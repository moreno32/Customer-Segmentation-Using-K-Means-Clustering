"""
Tests for clustering.py

This module contains test functions for the clustering utilities,
including optimal cluster determination and K-means functionality.
"""

import os
import numpy as np
import pandas as pd
import pytest
import sys
from pathlib import Path
import tempfile
import matplotlib.pyplot as plt

# Add parent directory to path to import app modules
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from app.utils.clustering import (
    determine_optimal_clusters_elbow,
    perform_kmeans_clustering,
    create_cluster_profile,
    plot_elbow_method,
    plot_silhouette_comparison
)

# Create sample data fixtures
@pytest.fixture
def sample_rfm_data():
    """Create a sample RFM dataset for testing clustering."""
    # Create 50 samples with 3 features (RFM)
    np.random.seed(42)  # For reproducibility
    
    # Generate three distinct clusters
    cluster1 = np.random.normal(loc=[5, 20, 500], scale=[1, 3, 50], size=(20, 3))
    cluster2 = np.random.normal(loc=[15, 5, 100], scale=[2, 1, 20], size=(15, 3))
    cluster3 = np.random.normal(loc=[30, 10, 300], scale=[3, 2, 40], size=(15, 3))
    
    # Combine clusters
    data = np.vstack([cluster1, cluster2, cluster3])
    
    # Create DataFrame
    df = pd.DataFrame(data, columns=['Recency', 'Frequency', 'Monetary'])
    
    # Add CustomerID
    df['CustomerID'] = [f'C{i:03d}' for i in range(1, len(df) + 1)]
    
    # Ensure all values are positive
    df['Recency'] = df['Recency'].abs()
    df['Frequency'] = df['Frequency'].abs()
    df['Monetary'] = df['Monetary'].abs()
    
    return df

def test_determine_optimal_clusters_elbow(sample_rfm_data):
    """Test the elbow method for determining optimal number of clusters."""
    # Get feature matrix
    X = sample_rfm_data[['Recency', 'Frequency', 'Monetary']].values
    
    # Run the elbow method
    k_values, inertia_values = determine_optimal_clusters_elbow(X, max_clusters=6)
    
    # Check output types and values
    assert isinstance(k_values, list)
    assert isinstance(inertia_values, list)
    assert len(k_values) == len(inertia_values)
    assert len(k_values) > 0
    assert k_values[0] == 1  # Should start from 1
    assert k_values[-1] <= 6  # Should not exceed max_clusters
    
    # Check if inertia is decreasing
    assert all(inertia_values[i] >= inertia_values[i+1] for i in range(len(inertia_values)-1))

def test_perform_kmeans_clustering(sample_rfm_data):
    """Test K-means clustering implementation."""
    # Get feature matrix
    X = sample_rfm_data[['Recency', 'Frequency', 'Monetary']].values
    
    # Perform clustering with 3 clusters
    cluster_labels, kmeans_model = perform_kmeans_clustering(X, n_clusters=3)
    
    # Check output
    assert cluster_labels is not None
    assert kmeans_model is not None
    assert len(cluster_labels) == len(sample_rfm_data)
    assert len(np.unique(cluster_labels)) == 3  # Should have 3 unique clusters
    
    # Test with different numbers of clusters
    for n_clusters in [2, 4]:
        cluster_labels, kmeans_model = perform_kmeans_clustering(X, n_clusters=n_clusters)
        assert len(np.unique(cluster_labels)) == n_clusters

def test_create_cluster_profile(sample_rfm_data):
    """Test creation of cluster profiles."""
    # Get feature matrix and perform clustering
    X = sample_rfm_data[['Recency', 'Frequency', 'Monetary']].values
    cluster_labels, _ = perform_kmeans_clustering(X, n_clusters=3)
    
    # Create normalized data
    normalized_rfm = sample_rfm_data.copy()
    
    # Create cluster profiles
    feature_names = ['Recency', 'Frequency', 'Monetary']
    profiles = create_cluster_profile(normalized_rfm, sample_rfm_data, cluster_labels, feature_names)
    
    # Check output
    assert profiles is not None
    assert isinstance(profiles, pd.DataFrame)
    assert len(profiles) == 3  # Should have 3 clusters
    assert all(col in profiles.columns for col in ['Recency', 'Frequency', 'Monetary', 'Size', 'Percentage'])
    
    # Check if percentages sum to approximately 100%
    assert abs(profiles['Percentage'].sum() - 100) < 1e-10

def test_plot_elbow_method():
    """Test elbow method plotting function."""
    # Sample data
    k_values = [1, 2, 3, 4, 5]
    inertia_values = [1000, 500, 300, 200, 150]
    
    # Create plot
    fig = plot_elbow_method(k_values, inertia_values)
    
    # Check output
    assert fig is not None
    
    # Basic test for plotly figure
    assert hasattr(fig, 'data')
    assert len(fig.data) > 0

def test_plot_silhouette_comparison(sample_rfm_data):
    """Test silhouette comparison plotting function."""
    # Get feature matrix
    X = sample_rfm_data[['Recency', 'Frequency', 'Monetary']].values
    
    # Create plot for range of clusters
    fig = plot_silhouette_comparison(X, 2, 4)
    
    # Check output
    assert fig is not None
    
    # Basic test for plotly figure
    assert hasattr(fig, 'data')
    assert len(fig.data) > 0

def test_error_handling():
    """Test error handling in clustering functions."""
    # Test with invalid input for determine_optimal_clusters_elbow
    with pytest.raises(Exception):  # Should raise an exception for invalid input
        determine_optimal_clusters_elbow(None, max_clusters=5)
    
    # Test with invalid number of clusters
    with pytest.raises(Exception):  # Should raise an exception for invalid cluster number
        X = np.random.rand(10, 3)
        perform_kmeans_clustering(X, n_clusters=0) 