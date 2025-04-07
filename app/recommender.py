"""
Recommender Module

Functions for generating product recommendations based on customer segments.
"""

import pandas as pd
import numpy as np

def get_cluster_recommendations(transaction_data, customer_clusters, cluster_id, top_n=10):
    """
    Get top product recommendations for a specific customer segment.
    
    Parameters:
    -----------
    transaction_data : pd.DataFrame
        Transaction data with product information
    customer_clusters : pd.DataFrame
        Mapping of customers to their assigned clusters
    cluster_id : int
        ID of the cluster to generate recommendations for
    top_n : int, optional
        Number of top products to recommend
    
    Returns:
    --------
    pd.DataFrame
        DataFrame with recommended products and their metrics
    """
    # Filter transactions for customers in the specified cluster
    cluster_customers = customer_clusters[customer_clusters['Cluster'] == cluster_id]['CustomerID'].tolist()
    
    if not cluster_customers:
        return pd.DataFrame()  # Empty DataFrame if no customers in cluster
    
    cluster_transactions = transaction_data[transaction_data['CustomerID'].isin(cluster_customers)]
    
    # Calculate product metrics
    product_metrics = cluster_transactions.groupby(['StockCode', 'Description']).agg({
        'Quantity': 'sum',
        'UnitPrice': 'mean',
        'InvoiceNo': 'nunique'
    }).reset_index()
    
    # Rename columns
    product_metrics.columns = ['StockCode', 'Description', 'TotalQuantity', 'UnitPrice', 'TransactionCount']
    
    # Calculate total revenue
    product_metrics['TotalRevenue'] = product_metrics['TotalQuantity'] * product_metrics['UnitPrice']
    
    # Sort by total revenue (or another relevant metric)
    product_metrics = product_metrics.sort_values('TotalRevenue', ascending=False)
    
    # Return top N products
    return product_metrics.head(top_n)

def calculate_product_affinity(transaction_data, customer_clusters):
    """
    Calculate product affinity scores for each cluster.
    
    Parameters:
    -----------
    transaction_data : pd.DataFrame
        Transaction data with product information
    customer_clusters : pd.DataFrame
        Mapping of customers to their assigned clusters
    
    Returns:
    --------
    pd.DataFrame
        DataFrame with product affinity scores by cluster
    """
    # Create merged dataset
    merged_data = transaction_data.merge(customer_clusters, on='CustomerID')
    
    # Group products by cluster and count occurrences
    cluster_product_counts = merged_data.groupby(['Cluster', 'StockCode', 'Description']).size().reset_index(name='ClusterCount')
    
    # Calculate total product counts
    product_counts = merged_data.groupby(['StockCode', 'Description']).size().reset_index(name='TotalCount')
    
    # Merge to get all counts in one DataFrame
    affinity_data = cluster_product_counts.merge(product_counts, on=['StockCode', 'Description'])
    
    # Calculate lift (affinity score)
    # Lift = (Cluster Product Count / Cluster Size) / (Total Product Count / Total Customers)
    cluster_sizes = customer_clusters['Cluster'].value_counts().reset_index()
    cluster_sizes.columns = ['Cluster', 'ClusterSize']
    
    total_customers = len(customer_clusters)
    
    # Merge cluster sizes
    affinity_data = affinity_data.merge(cluster_sizes, on='Cluster')
    
    # Calculate lift
    affinity_data['Lift'] = (affinity_data['ClusterCount'] / affinity_data['ClusterSize']) / (affinity_data['TotalCount'] / total_customers)
    
    # Sort by cluster and lift
    affinity_data = affinity_data.sort_values(['Cluster', 'Lift'], ascending=[True, False])
    
    return affinity_data

def get_customer_recommendations(customer_id, transaction_data, customer_clusters, top_n=5):
    """
    Get personalized product recommendations for a specific customer.
    
    Parameters:
    -----------
    customer_id : str
        ID of the customer to generate recommendations for
    transaction_data : pd.DataFrame
        Transaction data with product information
    customer_clusters : pd.DataFrame
        Mapping of customers to their assigned clusters
    top_n : int, optional
        Number of top products to recommend
    
    Returns:
    --------
    pd.DataFrame
        DataFrame with recommended products and their scores
    """
    # Get the customer's cluster
    customer_row = customer_clusters[customer_clusters['CustomerID'] == customer_id]
    if customer_row.empty:
        return pd.DataFrame()  # Customer not found
    
    cluster_id = customer_row.iloc[0]['Cluster']
    
    # Get products the customer has already purchased
    customer_transactions = transaction_data[transaction_data['CustomerID'] == customer_id]
    purchased_products = set(customer_transactions['StockCode'].unique())
    
    # Get top products for the cluster
    cluster_products = get_cluster_recommendations(
        transaction_data,
        customer_clusters,
        cluster_id,
        top_n=20  # Get more than needed to filter
    )
    
    # Filter out products the customer has already purchased
    new_products = cluster_products[~cluster_products['StockCode'].isin(purchased_products)]
    
    # Calculate a personalized score
    # Simple approach: use TotalRevenue as the score
    new_products['Score'] = new_products['TotalRevenue']
    
    # Return top N products
    return new_products[['StockCode', 'Description', 'UnitPrice', 'Score']].head(top_n)

def get_similar_customers(customer_id, customer_clusters, top_n=5):
    """
    Find similar customers based on cluster membership.
    
    Parameters:
    -----------
    customer_id : str
        ID of the customer to find similar customers for
    customer_clusters : pd.DataFrame
        Mapping of customers to their assigned clusters
    top_n : int, optional
        Number of similar customers to return
    
    Returns:
    --------
    list
        List of similar customer IDs
    """
    # Get the customer's cluster
    customer_row = customer_clusters[customer_clusters['CustomerID'] == customer_id]
    if customer_row.empty:
        return []  # Customer not found
    
    cluster_id = customer_row.iloc[0]['Cluster']
    
    # Get other customers in the same cluster
    similar_customers = customer_clusters[
        (customer_clusters['Cluster'] == cluster_id) & 
        (customer_clusters['CustomerID'] != customer_id)
    ]['CustomerID'].tolist()
    
    # Return top N similar customers
    return similar_customers[:top_n] 