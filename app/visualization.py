"""
Visualization Module

Functions for creating professional, monochromatic visualizations of customer data,
clusters, and recommendations.
"""

import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from utils import get_cluster_colors, get_monochromatic_color_sequence
from styles import COLORS, THEME

# Monochromatic color palette for visualizations
GRAYSCALE_COLORS = get_cluster_colors()

# Template for professional Plotly visualizations
PLOTLY_TEMPLATE = {
    "layout": {
        "font": {
            "family": THEME["typography"]["font_family"],
            "color": COLORS["text"]
        },
        "plot_bgcolor": COLORS["background"],
        "paper_bgcolor": COLORS["background"],
        "colorway": get_monochromatic_color_sequence(8),
        "xaxis": {
            "gridcolor": "#EEEEEE",
            "zerolinecolor": "#DDDDDD",
            "title": {"font": {"size": 14}},
        },
        "yaxis": {
            "gridcolor": "#EEEEEE",
            "zerolinecolor": "#DDDDDD",
            "title": {"font": {"size": 14}},
        },
        "legend": {
            "font": {"size": 12},
            "title": {"font": {"size": 14}}
        },
        "margin": {"t": 60, "b": 60, "l": 40, "r": 40}
    }
}

def plot_rfm_3d(rfm_data, cluster_labels=None):
    """
    Create a professional 3D scatter plot of RFM data with monochromatic styling.
    
    Parameters:
    -----------
    rfm_data : pd.DataFrame
        DataFrame with RFM features
    cluster_labels : np.ndarray, optional
        Cluster labels for each customer
    
    Returns:
    --------
    plotly.graph_objects.Figure
        3D scatter plot of RFM data
    """
    # Create copy of data to avoid modifying original
    plot_data = rfm_data.copy()
    
    # Apply log transformation to improve visualization
    plot_data['log_Monetary'] = np.log1p(plot_data['Monetary'])
    plot_data['log_Frequency'] = np.log1p(plot_data['Frequency'])
    plot_data['log_Recency'] = np.log1p(plot_data['Recency'])
    
    # Set up color, size, and hover information
    if cluster_labels is not None:
        # With clusters: use grayscale for each cluster
        plot_data['Cluster'] = cluster_labels
        unique_clusters = sorted(set(cluster_labels))
        n_clusters = len(unique_clusters)
        
        color_param = 'Cluster'
        color_discrete_map = {i: color for i, color in enumerate(get_monochromatic_color_sequence(n_clusters))}
        color_continuous_scale = None
        
        # Vary point size by monetary value
        plot_data['point_size'] = np.log1p(plot_data['Monetary']) + 3
        
        hover_data = {
            'CustomerID': True,
            'Recency': True,
            'Frequency': True,
            'Monetary': True,
            'Cluster': True,
            'point_size': False  # Hide from hover data
        }
        title = 'Customer Segments in RFM Space'
        legend_title = "Customer Segment"
    else:
        # Without clusters: use grayscale gradient by monetary value
        color_param = 'log_Monetary'
        color_discrete_map = None
        color_continuous_scale = px.colors.sequential.Greys
        
        # Constant point size
        plot_data['point_size'] = 4
        
        hover_data = {
            'CustomerID': True,
            'Recency': True,
            'Frequency': True,
            'Monetary': True,
            'point_size': False  # Hide from hover data
        }
        title = 'Customer Distribution in RFM Space'
        legend_title = None # No legend title needed if no clusters
    
    # Create the 3D scatter plot WITHOUT title initially
    fig = px.scatter_3d(
        plot_data,
        x='log_Recency',
        y='log_Frequency',
        z='log_Monetary',
        color=color_param,
        size='point_size',
        color_discrete_map=color_discrete_map,
        color_continuous_scale=color_continuous_scale,
        opacity=0.8,
        hover_data=hover_data,
        labels={
            'log_Recency': 'Recency (log)',
            'log_Frequency': 'Frequency (log)',
            'log_Monetary': 'Monetary (log)',
            'Cluster': 'Segment'
        }
        # title=title # Removed title from here
    )
    
    # Apply the base template layout
    fig.update_layout(**PLOTLY_TEMPLATE["layout"])
    
    # Apply specific updates for this plot, including title and legend title safely
    fig.update_layout(
        title_text=title, # Use title_text to set/update title
        scene=dict(
            xaxis=dict(title='Recency (days)', showgrid=True, gridcolor='#EEEEEE', zeroline=True, zerolinecolor='#DDDDDD'),
            yaxis=dict(title='Frequency (purchases)', showgrid=True, gridcolor='#EEEEEE', zeroline=True, zerolinecolor='#DDDDDD'),
            zaxis=dict(title='Monetary (amount)', showgrid=True, gridcolor='#EEEEEE', zeroline=True, zerolinecolor='#DDDDDD'),
            aspectmode='cube',
            camera=dict(up=dict(x=0, y=0, z=1), center=dict(x=0, y=0, z=0), eye=dict(x=1.5, y=1.5, z=1.5))
        ),
        legend_title_text=legend_title, # Use legend_title_text to set/update legend title
        height=600
    )
    
    # Ensure legend title is removed if it's None (e.g., no clusters)
    if legend_title is None:
        fig.update_layout(legend_title_text=None)
    
    return fig

def plot_elbow_method(k_values, inertia_values):
    """
    Create a professional plot of the Elbow Method with monochromatic styling.
    
    Parameters:
    -----------
    k_values : list
        List of k values (number of clusters)
    inertia_values : list
        Corresponding inertia values
    
    Returns:
    --------
    plotly.graph_objects.Figure
        Line plot of the Elbow Method
    """
    # Create the line plot
    fig = go.Figure()
    
    # Add inertia line
    fig.add_trace(go.Scatter(
        x=k_values,
        y=inertia_values,
        mode='lines+markers',
        name='Inertia',
        marker=dict(
            size=10, 
            color=COLORS['highlight'],
            line=dict(width=1, color=COLORS['background'])
        ),
        line=dict(width=2, color=COLORS['accent'])
    ))
    
    # Find the elbow point using second derivative
    if len(k_values) > 2:
        second_derivative = np.diff(np.diff(inertia_values))
        elbow_index = np.argmax(second_derivative) + 2
        elbow_x = k_values[elbow_index]
        elbow_y = inertia_values[elbow_index]
        
        # Add marker for elbow point
        fig.add_trace(go.Scatter(
            x=[elbow_x],
            y=[elbow_y],
            mode='markers',
            marker=dict(
                size=15, 
                color=COLORS['highlight'],
                symbol='star',
                line=dict(width=1, color=COLORS['background'])
            ),
            name=f'Optimal k={elbow_x}',
            hoverinfo='name'
        ))
    
    # Update layout with professional styling - removing duplicate title
    layout_updates = {
        "xaxis": dict(title='Number of Clusters (k)', tickmode='linear', showgrid=True, gridcolor='#EEEEEE'),
        "yaxis": dict(title='Inertia (Within-Cluster Sum of Squares)', showgrid=True, gridcolor='#EEEEEE'),
        "showlegend": True,
        "height": 500
    }
    # Apply template first, then specific updates, ensuring title is not duplicated
    fig.update_layout(**PLOTLY_TEMPLATE["layout"])
    fig.update_layout(**layout_updates)
    fig.update_layout(title_text='Optimal Number of Clusters (Elbow Method)') # Set title separately

    # Apply the base template layout first, excluding title settings from template
    base_layout = PLOTLY_TEMPLATE["layout"].copy()
    base_layout.pop('title', None) # Remove title from base to avoid conflict
    fig.update_layout(**base_layout)

    # Now apply specific updates for this plot
    fig.update_layout(
        title_text='Optimal Number of Clusters (Elbow Method)', # Set title explicitly here
        xaxis=dict(
            title='Number of Clusters (k)',
            tickmode='linear',
            showgrid=True,
            gridcolor='#EEEEEE'
        ),
        yaxis=dict(
            title='Inertia (Within-Cluster Sum of Squares)',
            showgrid=True,
            gridcolor='#EEEEEE'
        ),
        showlegend=True,
        height=500
    )
    
    return fig

def plot_cluster_profiles(cluster_profiles):
    """Creates a list of radar charts, one for each cluster profile."""
    # --- Determine features available in the profiles --- 
    # Exclude known non-feature columns
    non_feature_cols = ['Size', 'Percentage', 'Typical Customer', 'Action Suggestion'] 
    feature_cols = [col for col in cluster_profiles.columns if col not in non_feature_cols]
    
    if not feature_cols: # Check if any feature columns are left
        st.error("No valid feature columns found in cluster profiles for Radar Chart.")
        return []
    elif len(feature_cols) < 2: # Radar needs at least 2 axes
         st.info(f"Radar chart requires at least 2 features. Only found: {', '.join(feature_cols)}.")
         return []
        
    # if not all(col in cluster_profiles.columns for col in feature_cols):
    #     st.error("Cluster profiles DataFrame is missing required feature columns.")
    #     return [] # Return empty list

    # --- Normalization (Min-Max Scaling 0-1 PER FEATURE based on profile values) ---
    # Normalizes each feature independently based on the min/max of that feature across the cluster averages.
    # This stretches each axis to fill the 0-1 range, making shapes more distinct.
    normalized_profiles = cluster_profiles[feature_cols].copy()
    for col in feature_cols:
        min_val = normalized_profiles[col].min()
        max_val = normalized_profiles[col].max()
        range_val = max_val - min_val
        if range_val > 0:
            normalized_profiles[col] = (normalized_profiles[col] - min_val) / range_val
        else:
            normalized_profiles[col] = 0.5 # Assign middle value if all are the same
    normalized_profiles = normalized_profiles.fillna(0.5) # Handle potential NaNs
    
    radar_charts = []
    n_clusters = len(cluster_profiles)
    cluster_colors = get_monochromatic_color_sequence(n_clusters)
    
    # Create a separate radar chart for each cluster
    for i, (cluster_idx, row) in enumerate(normalized_profiles.iterrows()):
        fig = go.Figure()
        original_values = cluster_profiles.loc[cluster_idx, feature_cols]
        r_values = row[feature_cols].values 
        r_values = np.append(r_values, r_values[0]) # Close the polygon
        theta_labels = feature_cols + [feature_cols[0]] # Use dynamic features
        
        # Generate hover text dynamically
        hover_texts = [f"<b>{feat}:</b> {original_values.get(feat, 'N/A'):.1f}" for feat in feature_cols]
        hover_texts.append(hover_texts[0]) 
        hover_template = "%{customdata}<extra></extra>" 
        cluster_color = cluster_colors[i % len(cluster_colors)]
        
        # Add trace using dynamic theta_labels
        fig.add_trace(go.Scatterpolar(
            r=r_values,
            theta=theta_labels, 
            mode='lines+markers',
            name=f'Segment {cluster_idx}', 
            line=dict(color=cluster_color, width=3), # Thicker line
            marker=dict(size=8, color=cluster_color), # Bigger marker
            fill='toself',
            fillcolor=f'rgba({int(cluster_color[1:3], 16)}, {int(cluster_color[3:5], 16)}, {int(cluster_color[5:7], 16)}, 0.3)', # Slightly more opaque fill
            customdata=hover_texts, 
            hovertemplate=hover_template
        ))

        # --- Layout Updates ---
        base_layout = PLOTLY_TEMPLATE["layout"].copy()
        base_layout.pop('title', None)
        fig.update_layout(**base_layout)
        fig.update_layout(
            title_text=f'Profile: Segment {cluster_idx}',
            polar=dict(
                radialaxis=dict(visible=True, range=[0, 1], showticklabels=True, ticksuffix='', 
                              gridcolor="#e0e0e0"), # Lighter grid
                angularaxis=dict(tickfont_size=10, direction = "clockwise") # Added direction
            ),
            showlegend=False, 
            height=300, # Slightly smaller to fit more per row if needed
            margin=dict(l=40, r=40, t=70, b=40) 
        )
        
        radar_charts.append(fig)
        
    return radar_charts

def plot_recommendations(recommendations):
    """
    Create a professional bar chart of product recommendations with monochromatic styling.
    
    Parameters:
    -----------
    recommendations : pd.DataFrame
        DataFrame with recommended products
    
    Returns:
    --------
    plotly.graph_objects.Figure
        Bar chart of product recommendations
    """
    # Create a copy and sort by score column
    if 'Score' in recommendations.columns:
        plot_data = recommendations.sort_values('Score', ascending=False)
        y_column = 'Score'
        title = 'Personalized Product Recommendations'
    elif 'TotalRevenue' in recommendations.columns:
        plot_data = recommendations.sort_values('TotalRevenue', ascending=False)
        y_column = 'TotalRevenue'
        title = 'Top Products for Selected Segment'
    else:
        # Default to the first numeric column
        numeric_cols = recommendations.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) > 0:
            y_column = numeric_cols[0]
            plot_data = recommendations.sort_values(y_column, ascending=False)
            title = 'Product Recommendations'
        else:
            return None
    
    # Truncate long product descriptions
    if 'Description' in plot_data.columns:
        plot_data['Short_Description'] = plot_data['Description'].apply(
            lambda x: x[:30] + '...' if len(x) > 30 else x
        )
        x_column = 'Short_Description'
    else:
        x_column = 'StockCode'
    
    # Create horizontal bar chart with monochromatic color scale
    plot_data = plot_data.head(10).copy()
    plot_data['index'] = range(len(plot_data))  # Add index for consistent ordering
    
    fig = go.Figure()
    
    # Add bars
    fig.add_trace(go.Bar(
        x=plot_data[y_column],
        y=plot_data[x_column],
        orientation='h',
        marker=dict(
            color=plot_data[y_column],
            colorscale=px.colors.sequential.Greys,
            line=dict(width=0)
        ),
        hovertemplate='<b>%{y}</b><br>Score: %{x:.2f}<extra></extra>'
    ))
    
    # Update layout with professional styling
    fig.update_layout(**PLOTLY_TEMPLATE["layout"])

    # Apply the base template layout first, excluding title settings from template
    base_layout = PLOTLY_TEMPLATE["layout"].copy()
    base_layout.pop('title', None) # Remove title from base to avoid conflict
    fig.update_layout(**base_layout)

    # Now apply specific updates for this plot
    fig.update_layout(
        xaxis=dict(title=y_column.replace('_', ' ').title(), showgrid=True, gridcolor='#EEEEEE'), # Dynamic X axis title
        yaxis=dict(title='Product', showgrid=False, autorange="reversed"),
        height=500,
        margin=dict(l=150, r=20, t=60, b=40) # Increased left margin for labels
    )
    fig.update_layout(title_text=title) # Set title explicitly here
    
    return fig 