"""
Tests for data_processor.py

This module contains test functions for the data processing utilities,
including data loading, cleaning, and feature engineering.
"""

import os
import pandas as pd
import numpy as np
import pytest
from pathlib import Path
import tempfile
import sys

# Add parent directory to path to import app modules
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from app.utils.data_processor import (
    load_data,
    clean_data,
    create_rfm_features,
    normalize_rfm_features
)

# Create test data
@pytest.fixture
def sample_transaction_data():
    """Create a sample transaction dataset for testing."""
    return pd.DataFrame({
        'InvoiceNo': ['A001', 'A001', 'A002', 'A003', 'A004', 'A004', 'A005'],
        'StockCode': ['P001', 'P002', 'P001', 'P003', 'P002', 'P004', 'P005'],
        'Description': ['Product 1', 'Product 2', 'Product 1', 'Product 3', 'Product 2', 'Product 4', 'Product 5'],
        'Quantity': [2, 1, 3, 5, 1, 3, -2],  # Note: negative quantity for testing
        'InvoiceDate': pd.to_datetime(['2020-01-15', '2020-01-15', '2020-02-20', '2020-03-10', 
                                       '2020-04-05', '2020-04-05', '2020-05-01']),
        'UnitPrice': [10.0, 15.0, 10.0, 8.0, 15.0, 20.0, 5.0],
        'CustomerID': ['C001', 'C001', 'C002', 'C001', 'C003', 'C003', 'C001']
    })

@pytest.fixture
def sample_csv_file():
    """Create a temporary CSV file with sample data for testing."""
    with tempfile.NamedTemporaryFile(suffix='.csv', delete=False) as tmp:
        df = pd.DataFrame({
            'InvoiceNo': ['A001', 'A002'],
            'StockCode': ['P001', 'P002'],
            'Description': ['Product 1', 'Product 2'],
            'Quantity': [2, 3],
            'InvoiceDate': ['2020-01-15', '2020-02-20'],
            'UnitPrice': [10.0, 15.0],
            'CustomerID': ['C001', 'C002']
        })
        df.to_csv(tmp.name, index=False)
        return tmp.name

def test_load_data(sample_csv_file):
    """Test loading data from a CSV file."""
    # Test successful load
    df = load_data(sample_csv_file)
    assert df is not None
    assert isinstance(df, pd.DataFrame)
    assert df.shape[0] > 0
    assert 'InvoiceDate' in df.columns
    assert pd.api.types.is_datetime64_dtype(df['InvoiceDate'])
    
    # Test with non-existent file
    df_none = load_data('non_existent_file.csv')
    assert df_none is None
    
    # Clean up
    os.remove(sample_csv_file)

def test_clean_data(sample_transaction_data):
    """Test data cleaning functionality."""
    # Create copy with some problematic data
    df = sample_transaction_data.copy()
    
    # Add a row with missing CustomerID
    missing_customer = df.iloc[0].copy()
    missing_customer['CustomerID'] = np.nan
    df = pd.concat([df, pd.DataFrame([missing_customer])], ignore_index=True)
    
    # Add a row with zero quantity
    zero_quantity = df.iloc[0].copy()
    zero_quantity['Quantity'] = 0
    df = pd.concat([df, pd.DataFrame([zero_quantity])], ignore_index=True)
    
    # Add duplicate row
    df = pd.concat([df, pd.DataFrame([df.iloc[0]])], ignore_index=True)
    
    # Clean the data
    cleaned_df = clean_data(df)
    
    # Verify results
    assert cleaned_df is not None
    assert isinstance(cleaned_df, pd.DataFrame)
    assert cleaned_df.shape[0] < df.shape[0]  # Should have removed problematic rows
    assert not cleaned_df['CustomerID'].isna().any()  # No missing CustomerIDs
    assert (cleaned_df['Quantity'] > 0).all()  # No non-positive quantities
    assert 'TotalPrice' in cleaned_df.columns  # Added TotalPrice column
    
    # Check if rows with negative quantities were removed
    assert not (cleaned_df['Quantity'] < 0).any()
    
    # Check if duplicates were removed
    assert cleaned_df.shape[0] == cleaned_df.drop_duplicates().shape[0]

def test_create_rfm_features(sample_transaction_data):
    """Test RFM feature creation."""
    # Clean the data first
    df = clean_data(sample_transaction_data)
    
    # Create RFM features without specifying reference date
    rfm_df = create_rfm_features(df)
    
    # Verify results
    assert rfm_df is not None
    assert isinstance(rfm_df, pd.DataFrame)
    assert 'CustomerID' in rfm_df.columns
    assert 'Recency' in rfm_df.columns
    assert 'Frequency' in rfm_df.columns
    assert 'Monetary' in rfm_df.columns
    
    # Basic properties checks
    assert (rfm_df['Recency'] >= 0).all()  # Recency should be non-negative
    assert (rfm_df['Frequency'] > 0).all()  # Frequency should be positive
    assert (rfm_df['Monetary'] > 0).all()  # Monetary should be positive
    
    # Test with custom reference date
    ref_date = pd.to_datetime('2020-06-01')
    rfm_df_custom = create_rfm_features(df, ref_date)
    
    # Verify with custom date
    assert rfm_df_custom is not None
    assert (rfm_df_custom['Recency'] >= 0).all()
    
    # Check if reference date affected recency calculation
    # Recency with later reference date should be greater than or equal to recency with earlier date
    c1 = rfm_df.set_index('CustomerID').loc['C001', 'Recency']
    c1_custom = rfm_df_custom.set_index('CustomerID').loc['C001', 'Recency']
    assert c1_custom >= c1

def test_normalize_rfm_features(sample_transaction_data):
    """Test normalization of RFM features."""
    # Prepare RFM data
    df = clean_data(sample_transaction_data)
    rfm_df = create_rfm_features(df)
    
    # Normalize features
    normalized_df, scaler = normalize_rfm_features(rfm_df)
    
    # Verify results
    assert normalized_df is not None
    assert scaler is not None
    assert isinstance(normalized_df, pd.DataFrame)
    assert normalized_df.shape == rfm_df.shape
    
    # Check if the data was actually normalized
    features = ['Recency', 'Frequency', 'Monetary']
    for feature in features:
        assert normalized_df[feature].mean() != rfm_df[feature].mean()
    
    # Test with provided scaler
    normalized_df2, _ = normalize_rfm_features(rfm_df, scaler)
    
    # Verify that using the same scaler gives same results
    pd.testing.assert_frame_equal(normalized_df[features], normalized_df2[features])
    
    # Test saving scaler
    with tempfile.TemporaryDirectory() as tmpdir:
        save_path = os.path.join(tmpdir, 'scaler.pkl')
        _, _ = normalize_rfm_features(rfm_df, save_path=save_path)
        assert os.path.exists(save_path)

def test_error_cases():
    """Test error handling in the data processing functions."""
    # Test with invalid dataframe for create_rfm_features
    invalid_df = pd.DataFrame({'A': [1, 2], 'B': [3, 4]})
    
    # This should handle the error and return None
    result = create_rfm_features(invalid_df)
    assert result is None 