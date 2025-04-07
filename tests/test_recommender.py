"""
Tests for recommender.py

This module contains test functions for the recommendation system utilities,
including product recommendations and affinity calculations.
"""

import os
import pandas as pd
import numpy as np
import pytest
import sys
from pathlib import Path

# Add parent directory to path to import app modules
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from app.utils.recommender import (
    get_top_products_by_cluster,
    calculate_product_affinity_by_cluster,
    get_product_recommendations_for_customer,
    visualize_recommendations
)

# Create sample data fixtures
@pytest.fixture
def sample_transaction_data():
    """Create a sample transaction dataset for testing recommendations."""
    # Create sample transaction data with 3 customers and 4 products
    return pd.DataFrame({
        'InvoiceNo': ['A001', 'A001', 'A002', 'A002', 'A003', 'A004', 'A005', 'A006', 'A007', 'A008'],
        'StockCode': ['P001', 'P002', 'P001', 'P003', 'P002', 'P004', 'P001', 'P003', 'P002', 'P004'],
        'Description': ['Product 1', 'Product 2', 'Product 1', 'Product 3', 'Product 2', 
                        'Product 4', 'Product 1', 'Product 3', 'Product 2', 'Product 4'],
        'Quantity': [2, 1, 1, 3, 2, 1, 2, 1, 3, 2],
        'UnitPrice': [10.0, 15.0, 10.0, 20.0, 15.0, 30.0, 10.0, 20.0, 15.0, 30.0],
        'CustomerID': ['C001', 'C001', 'C002', 'C002', 'C001', 'C003', 'C002', 'C003', 'C001', 'C002'],
        'TotalPrice': [20.0, 15.0, 10.0, 60.0, 30.0, 30.0, 20.0, 20.0, 45.0, 60.0]  # Quantity * UnitPrice
    })

@pytest.fixture
def sample_customer_clusters():
    """Create a sample customer clustering result."""
    return pd.DataFrame({
        'CustomerID': ['C001', 'C002', 'C003'],
        'Cluster': [0, 1, 2]  # Each customer in a different cluster for testing
    })

def test_get_top_products_by_cluster(sample_transaction_data, sample_customer_clusters):
    """Test getting top products for a specific cluster."""
    # Get top products for cluster 0 (customer C001)
    top_products = get_top_products_by_cluster(
        sample_transaction_data,
        sample_customer_clusters,
        cluster_id=0
    )
    
    # Check output
    assert top_products is not None
    assert isinstance(top_products, pd.DataFrame)
    assert not top_products.empty
    
    # Check if result has the expected columns
    expected_columns = ['StockCode', 'Description', 'TotalQuantity', 'UnitPrice', 'TotalRevenue', 'TransactionCount']
    assert all(col in top_products.columns for col in expected_columns)
    
    # Verify top product for cluster 0 is correct (based on our fixture data)
    assert 'P002' in top_products['StockCode'].values  # Customer C001 buys a lot of P002
    
    # Test other clusters
    for cluster_id in [1, 2]:
        top_prods = get_top_products_by_cluster(
            sample_transaction_data,
            sample_customer_clusters,
            cluster_id=cluster_id
        )
        assert top_prods is not None
        assert not top_prods.empty

def test_calculate_product_affinity_by_cluster(sample_transaction_data, sample_customer_clusters):
    """Test calculation of product affinity scores by cluster."""
    # Calculate affinities
    affinities = calculate_product_affinity_by_cluster(
        sample_transaction_data,
        sample_customer_clusters
    )
    
    # Check output
    assert affinities is not None
    assert isinstance(affinities, pd.DataFrame)
    assert not affinities.empty
    
    # Check if result has the expected columns
    expected_columns = ['Cluster', 'StockCode', 'Description', 'ClusterCount', 'TotalCount', 'Lift']
    assert all(col in affinities.columns for col in expected_columns)
    
    # Check if all clusters are represented
    assert set(affinities['Cluster'].unique()) == set(sample_customer_clusters['Cluster'].unique())
    
    # Check if lift scores are calculated correctly (should be > 0)
    assert (affinities['Lift'] > 0).all()

def test_get_product_recommendations_for_customer(sample_transaction_data, sample_customer_clusters):
    """Test getting personalized recommendations for a specific customer."""
    # Calculate affinities first (prerequisite)
    affinities = calculate_product_affinity_by_cluster(
        sample_transaction_data,
        sample_customer_clusters
    )
    
    # Get recommendations for customer C001
    recommendations = get_product_recommendations_for_customer(
        customer_id='C001',
        transaction_data=sample_transaction_data,
        customer_clusters=sample_customer_clusters,
        product_affinities=affinities
    )
    
    # Check output
    assert recommendations is not None
    assert isinstance(recommendations, pd.DataFrame)
    assert not recommendations.empty
    
    # Verify we don't recommend products the customer has already purchased frequently
    customer_purchases = sample_transaction_data[sample_transaction_data['CustomerID'] == 'C001']
    frequently_purchased = customer_purchases.groupby('StockCode')['Quantity'].sum()
    frequently_purchased = frequently_purchased[frequently_purchased > 2].index.tolist()
    
    if frequently_purchased:
        # Check that top recommendations don't include frequently purchased items
        # (this logic may need adjustment based on your exact implementation)
        top_recommendation = recommendations.iloc[0]['StockCode']
        assert top_recommendation not in frequently_purchased

def test_visualize_recommendations():
    """Test visualization of recommendations."""
    # Create sample recommendation data
    recommendations = pd.DataFrame({
        'StockCode': ['P001', 'P002', 'P003'],
        'Description': ['Product 1', 'Product 2', 'Product 3'],
        'Lift': [2.5, 1.8, 1.2]
    })
    
    # Create visualization
    fig = visualize_recommendations(recommendations)
    
    # Check output
    assert fig is not None
    
    # Basic test for plotly figure
    assert hasattr(fig, 'data')
    assert len(fig.data) > 0

def test_error_handling(sample_transaction_data, sample_customer_clusters):
    """Test error handling in recommendation functions."""
    # Test with invalid customer ID
    recommendations = get_product_recommendations_for_customer(
        customer_id='INVALID',
        transaction_data=sample_transaction_data,
        customer_clusters=sample_customer_clusters,
        product_affinities=pd.DataFrame()  # Empty affinities
    )
    
    # Should return empty DataFrame, not error
    assert isinstance(recommendations, pd.DataFrame)
    assert recommendations.empty
    
    # Test with invalid cluster ID
    top_products = get_top_products_by_cluster(
        sample_transaction_data,
        sample_customer_clusters,
        cluster_id=999  # Non-existent cluster
    )
    
    # Should return empty DataFrame, not error
    assert isinstance(top_products, pd.DataFrame)
    assert top_products.empty 