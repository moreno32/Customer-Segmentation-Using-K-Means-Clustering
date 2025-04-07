"""
Clustering Module

Functions for customer segmentation using K-means clustering.
"""

import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

def determine_optimal_clusters(data, max_clusters=8):
    """
    Determine the optimal number of clusters using the Elbow Method,
    adjusting max_clusters based on data size.
    
    Parameters:
    -----------
    data : np.ndarray
        Feature matrix for clustering (n_samples, n_features)
    max_clusters : int, optional
        Maximum number of clusters to evaluate
    
    Returns:
    --------
    tuple (list, list)
        Lists of k values and corresponding inertia values
    """
    n_samples = data.shape[0]
    
    # Adjust max_clusters if n_samples is too small
    # K-Means requires n_samples >= n_clusters
    # We also need at least 2 clusters for meaningful analysis
    adjusted_max_clusters = min(max_clusters, n_samples)
    
    if adjusted_max_clusters < 2:
        print(f"Warning: Not enough samples ({n_samples}) to perform clustering beyond K=1.")
        return [1], [KMeans(n_clusters=1, random_state=42, n_init=10).fit(data).inertia_]
        
    # Ensure k_values only go up to adjusted_max_clusters
    k_values = list(range(1, adjusted_max_clusters + 1))
    inertia_values = []
    
    print(f"Calculating inertia for K values: {k_values} (adjusted from max_clusters={max_clusters} due to n_samples={n_samples})")
    
    for k in k_values:
        # Handle potential warning if n_init > 1 and k > n_samples (though we adjusted k_values)
        current_n_init = 10 if n_samples >= k else 1 
        model = KMeans(n_clusters=k, random_state=42, n_init=current_n_init)
        model.fit(data)
        inertia_values.append(model.inertia_)
    
    return k_values, inertia_values

def perform_clustering(data, n_clusters=3):
    """
    Perform K-means clustering on customer data.
    
    Parameters:
    -----------
    data : np.ndarray
        Feature matrix for clustering
    n_clusters : int, optional
        Number of clusters to create
    
    Returns:
    --------
    tuple (np.ndarray, KMeans)
        Cluster labels and the trained KMeans model
    """
    model = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    labels = model.fit_predict(data)
    
    return labels, model

def create_cluster_profiles(normalized_data, original_data, cluster_labels, feature_names):
    """
    Create profiles for each cluster showing average feature values.
    
    Parameters:
    -----------
    normalized_data : pd.DataFrame
        Normalized data used for clustering
    original_data : pd.DataFrame
        Original data with unnormalized values
    cluster_labels : np.ndarray
        Cluster assignments from K-means
    feature_names : list
        Names of features used for clustering
    
    Returns:
    --------
    pd.DataFrame
        DataFrame with cluster profiles
    """
    # Create a copy of the data with cluster labels
    data_with_clusters = original_data.copy()
    data_with_clusters['Cluster'] = cluster_labels
    
    # Calculate average values for each feature in each cluster
    profiles = data_with_clusters.groupby('Cluster')[feature_names].mean()
    
    # Add cluster size information
    cluster_sizes = pd.DataFrame(data_with_clusters['Cluster'].value_counts()).reset_index()
    cluster_sizes.columns = ['Cluster', 'Size']
    cluster_sizes['Percentage'] = 100 * cluster_sizes['Size'] / len(data_with_clusters)
    
    # Merge size information with profiles
    profiles = profiles.merge(cluster_sizes, left_index=True, right_on='Cluster').set_index('Cluster')
    
    return profiles

def evaluate_clusters(data, cluster_labels):
    """
    Evaluate clustering quality using silhouette score.
    
    Parameters:
    -----------
    data : np.ndarray
        Feature matrix used for clustering
    cluster_labels : np.ndarray
        Cluster assignments from K-means
    
    Returns:
    --------
    float
        Silhouette score (between -1 and 1, higher is better)
    """
    if len(np.unique(cluster_labels)) < 2:
        return 0  # Silhouette score requires at least 2 clusters
    
    score = silhouette_score(data, cluster_labels)
    return score

def get_cluster_for_new_customer(customer_features, scaler, model):
    """
    Predict cluster for a new customer.
    
    Parameters:
    -----------
    customer_features : np.ndarray
        Customer RFM features (Recency, Frequency, Monetary)
    scaler : StandardScaler
        Scaler used to normalize the training data
    model : KMeans
        Trained KMeans model
    
    Returns:
    --------
    int
        Predicted cluster
    """
    # Normalize features
    normalized_features = scaler.transform(customer_features.reshape(1, -1))
    
    # Predict cluster
    cluster = model.predict(normalized_features)[0]
    
    return cluster 