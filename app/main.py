"""
E-commerce Recommendation System

Main Streamlit application refactored for clarity and reliable navigation.
"""

import streamlit as st
import pandas as pd
import numpy as np
import os
from pathlib import Path
import plotly.express as px

# Import functionality from other modules
# Ensure these imports are correct and modules exist
try:
    from data import load_data, clean_data, create_rfm_features, normalize_data
    from clustering import determine_optimal_clusters, perform_clustering, create_cluster_profiles
    from recommender import get_cluster_recommendations, get_customer_recommendations
    # Import the template directly
    from visualization import plot_rfm_3d, plot_elbow_method, plot_cluster_profiles, plot_recommendations, PLOTLY_TEMPLATE
    from utils import setup_page, init_session_state, display_error, display_success, format_currency, format_number, add_divider
    from styles import COLORS # Assuming styles contains COLORS dict
except ImportError as e:
    st.error(f"Error importing module: {e}. Please ensure all modules (data, clustering, etc.) exist.")
    st.stop() # Stop execution if modules are missing

# Application workflow steps
WORKFLOW_STEPS = [
    "1. Data Selection",
    "2. Data Exploration",
    "3. Customer Segmentation",
    "4. Recommendations",
    "5. Interactive Demo"
]

# --- Session State Initialization ---
def initialize_app_state():
    """Initializes session state variables if they don't exist."""
    defaults = {
        "current_page": "1. Data Selection",
        "original_data": None,
        "cleaned_data": None,
        "rfm_data": None,
        "normalized_data": None,
        "scaler": None,
        "data_processed": False,
        "n_clusters": 3,
        "ref_date": pd.to_datetime('2011-12-10').date(), # Default reference date
        "k_values": None,
        "inertia": None,
        "optimal_clusters": None,
        "cluster_labels": None,
        "cluster_model": None,
        "cluster_profiles": None,
        "customer_clusters": None,
        "segment_recommendations": None,
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value

# --- Page Display Functions ---

# Helper function for segment descriptions (moved globally)
def get_segment_description(profile):
    # Check if RFM data is available in session state
    if st.session_state.rfm_data is None:
        return "(Data not available)"
    try:
        # Use pd.Series.quantile for robustness
        quantiles = {
            'Recency': st.session_state.rfm_data['Recency'].quantile([0.33, 0.66]),
            'Frequency': st.session_state.rfm_data['Frequency'].quantile([0.33, 0.66]),
            'Monetary': st.session_state.rfm_data['Monetary'].quantile([0.33, 0.66])
        }
        
        # Safely get values, default to NaN if not present
        r = profile.get('Recency', np.nan)
        f = profile.get('Frequency', np.nan)
        m = profile.get('Monetary', np.nan)
        
        desc = []
        # Add descriptions only if the feature was used (value is not NaN)
        if not pd.isna(r):
            if r <= quantiles['Recency'][0.33]: desc.append("Recent")
            elif r > quantiles['Recency'][0.66]: desc.append("Lapsed")
            # else: desc.append("Occasional") 
        
        if not pd.isna(f):
            if f >= quantiles['Frequency'][0.66]: desc.append("Frequent")
            elif f < quantiles['Frequency'][0.33]: desc.append("Infrequent")
            
        if not pd.isna(m):
            if m >= quantiles['Monetary'][0.66]: desc.append("High Value")
            elif m < quantiles['Monetary'][0.33]: desc.append("Low Value")
                 
        # Combine into meaningful labels 
        if "Recent" in desc and "Frequent" in desc and "High Value" in desc: return "⭐ VIP Customers" 
        if "Lapsed" in desc and "Infrequent" in desc: return "⚠️ At Risk / Churned"    
        if "Recent" in desc and "Infrequent" in desc: return "👋 New / Potential" 
             
        return ", ".join(desc) if desc else "General Segment"
    except Exception as e:
        return "(Description Error)"

# Home page with data loading
def display_home_page():
    """Displays the Data Selection page (Step 1) using a more direct flow."""
    st.header("1. 💾 Load & Prepare Data")
    st.markdown("**Goal:** Load customer purchase history to begin analysis.")

    # --- Step A: Choose Data Source ---
    st.subheader("A. Choose Data Source")
    data_option = st.radio(
        "Source Type:",
        ["Use Example Data", "Upload CSV File"],
        key="data_source_selector_final",
        horizontal=True,
        label_visibility="collapsed"
    )

    # --- Step B: Load Data (Conditional based on selection) ---
    st.subheader("B. Load Your Data")
    data_loaded_successfully = False # Flag to track if valid data is in state
    if "user_uploaded_data" not in st.session_state: st.session_state.user_uploaded_data = None
    if "example_data_loaded" not in st.session_state: st.session_state.example_data_loaded = None

    if data_option == "Upload CSV File":
        uploaded_file = st.file_uploader(
            "Upload CSV (must contain CustomerID, InvoiceDate, Quantity, UnitPrice)", 
            type="csv",
            key="file_uploader_direct"
        )
        # Instant processing/validation on upload
        if uploaded_file is not None:
            # Check if it's a new file upload instance
            if uploaded_file != st.session_state.get("_uploaded_file_instance"): 
                 st.session_state._uploaded_file_instance = uploaded_file # Store instance
                 st.session_state.user_uploaded_data = None # Reset data on new upload
                 with st.spinner(f"Reading and validating '{uploaded_file.name}'..."):
                    try:
                        df_temp = pd.read_csv(uploaded_file, encoding='ISO-8859-1')
                        required_cols = ['CustomerID', 'InvoiceDate', 'Quantity', 'UnitPrice']
                        missing_cols = [col for col in required_cols if col not in df_temp.columns]
                        if not missing_cols:
                            df_temp['InvoiceDate'] = pd.to_datetime(df_temp['InvoiceDate'])
                            # Validation successful, store in state
                            st.session_state.user_uploaded_data = df_temp
                            st.session_state.example_data_loaded = None # Clear example if upload succeeds
                            st.success(f"✅ File '{uploaded_file.name}' validated! Preview below.")
                        else:
                            st.error(f"❌ Error: Uploaded file missing columns: {', '.join(missing_cols)}")
                            st.session_state.user_uploaded_data = None
                    except Exception as e:
                        st.error(f"❌ Error reading/validating file: {e}")
                        st.session_state.user_uploaded_data = None
        
        # Show preview if upload is valid
        if st.session_state.user_uploaded_data is not None:
            with st.expander("Preview Uploaded Data (First 5 Rows)", expanded=True):
                st.dataframe(st.session_state.user_uploaded_data.head(), use_container_width=True)
            data_loaded_successfully = True
        
    else: # Use Example Data
        st.caption("Using example: UK Online Retailer data (2010-2011).")
        if st.button("Load Example Data", key="load_example_direct_btn"):
            try:
                with st.spinner("Loading example data..."):
                    df = load_data() 
                    if df is not None:
                        st.session_state.example_data_loaded = df # Store example data separately
                        st.session_state.user_uploaded_data = None # Clear upload 
                        st.success("✅ Example data loaded! Preview below.")
                    else:
                        st.error("Failed to load example data.")
            except Exception as e:
                st.error(f"Error loading example data: {e}")
        
        # Show preview if example data is loaded
        if st.session_state.example_data_loaded is not None:
            with st.expander("Preview Example Data (First 5 Rows)", expanded=True):
                st.dataframe(st.session_state.example_data_loaded.head(), use_container_width=True)
            data_loaded_successfully = True

    # --- Step C: Set Parameters (Always visible, but maybe conceptually after loading) ---
    st.write("---")
    st.subheader("C. Set Analysis Parameters")
    col1_params, col2_params = st.columns(2)
    with col1_params:
        k_value = st.session_state.get("n_clusters", 3)
        st.session_state.n_clusters = st.number_input(
            "Number of Customer Groups (K)", 
            min_value=2, max_value=10, 
            value=k_value, step=1,
            key="k_selector_direct",
            help="How many distinct customer groups? (Usually 3-6)"
        )
    with col2_params:
        ref_date_value = st.session_state.get("ref_date", pd.to_datetime('2011-12-10').date())
        # Ensure we use a datetime object for the input value if loading from state
        try: 
             current_ref_date = pd.to_datetime(ref_date_value)
        except: 
             current_ref_date = pd.to_datetime('2011-12-10') # Fallback
             
        selected_ref_date = st.date_input(
            "Reference Date for Analysis",
            value=current_ref_date, 
            key="ref_date_direct",
            help="Date for calculating recency (day after last transaction)."
        )
        st.session_state.ref_date = pd.to_datetime(selected_ref_date).date() # Store date object

    # --- Step D: Process Data (Button enabled only when ready) ---
    st.write("---")
    st.subheader("D. Start Analysis")
    
    # Determine which data to use for processing
    data_to_process = None
    if st.session_state.user_uploaded_data is not None:
        data_to_process = st.session_state.user_uploaded_data
        data_source_name = "uploaded file"
    elif st.session_state.example_data_loaded is not None:
        data_to_process = st.session_state.example_data_loaded
        data_source_name = "example data"
        
    process_col1, process_col2, process_col3 = st.columns([1,2,1])
    with process_col2:
        # Enable button only if data is loaded
        if st.button("**📊 Start RFM Calculation & Segmentation Prep**", key="process_data_direct_btn", 
                     use_container_width=True, type="primary", disabled=(data_to_process is None)):
             
             # Store the data source being processed into the main session state variable
             st.session_state.original_data = data_to_process
             
             with st.spinner("Processing... (Cleaning, Calculating RFM, Normalizing)"):
                try:
                    # --- Determine Reference Date --- 
                    is_synthetic = False
                    if 'Country' in st.session_state.original_data.columns: 
                         if st.session_state.original_data['Country'].iloc[0] == 'Synthlandia V2':
                             is_synthetic = True
                    
                    if is_synthetic:
                        data_max_date = st.session_state.original_data['InvoiceDate'].max()
                        ref_datetime = data_max_date + pd.Timedelta(days=1)
                        st.session_state.ref_date = ref_datetime.date() # Update state ref_date for synthetic
                        print(f"Using auto-detected reference date for synthetic data: {ref_datetime.date()}")
                    else:
                        ref_datetime = pd.to_datetime(st.session_state.ref_date)
                        print(f"Using reference date from UI: {ref_datetime.date()}")
                    
                    # --- Run Processing Steps --- 
                    st.session_state.cleaned_data = clean_data(st.session_state.original_data)
                    st.session_state.rfm_data = create_rfm_features(st.session_state.cleaned_data, ref_datetime)
                    st.session_state.normalized_data, st.session_state.scaler = normalize_data(st.session_state.rfm_data)
                    
                    st.session_state.data_processed = True 
                    st.success(f"Data from '{data_source_name}' processed! Ready for Exploration.")
                    st.balloons()
                    
                    import time
                    time.sleep(1.5) 
                    
                    st.session_state.current_page = "2. Data Exploration"
                    st.experimental_rerun()
                    
                except Exception as e:
                    st.error(f"Error during data processing: {e}")
                    st.session_state.data_processed = False 
        elif data_to_process is None:
             st.info("☝️ Load data (Step B) and set parameters (Step C) to enable analysis.")

def display_data_exploration():
    """Displays the Data Exploration page (Step 2) with improved visualizations and explanations."""
    st.header("2. 探索 Exploring Customer Behavior (RFM)")
    st.subheader("Understanding Key Purchase Patterns")
    if not st.session_state.get('data_processed', False):
        st.warning("Data has not been processed yet. Please go back to '1. Data Selection'.")
        if st.button("Go back to Data Selection", key="back_to_sel_expl_1"):
             st.session_state.current_page = "1. Data Selection"
             st.experimental_rerun()
        return

    st.markdown("**Goal:** Analyze customer behavior using Recency, Frequency, and Monetary value (RFM) to understand your customer base.")
    st.markdown("Let's analyze your customers based on:")
    st.markdown("- **Recency (R):** How recently did they buy? *(Lower days = Better)*")
    st.markdown("- **Frequency (F):** How often do they buy? *(Higher count = Better)*")
    st.markdown("- **Monetary Value (M):** How much do they spend? *(Higher value = Better)*")
    
    st.write("---")
    st.subheader("📊 Overall Customer Snapshot (Averages)")
    st.caption("Average values across all customers in the dataset.")
    
    if st.session_state.rfm_data is not None:
        col1, col2, col3 = st.columns(3)
        avg_recency = st.session_state.rfm_data['Recency'].mean()
        avg_frequency = st.session_state.rfm_data['Frequency'].mean()
        avg_monetary = st.session_state.rfm_data['Monetary'].mean()
        
        col1.metric("Avg. Recency", f"{avg_recency:.0f} days", delta=None, help="On average, customers last purchased this many days ago.")
        col2.metric("Avg. Frequency", f"{avg_frequency:.1f} orders", delta=None, help="On average, customers have placed this many orders.")
        col3.metric("Avg. Monetary Value", f"{format_currency(avg_monetary)}", delta=None, help="On average, customers have spent this much.")
    else:
        st.info("RFM data not available.")

    st.write("---")
    st.subheader("📊 Distributional Patterns (Histograms)")
    st.markdown("**How are customers spread across each RFM dimension?** Understanding the distribution helps identify common behaviors.")
    if st.session_state.rfm_data is not None:
        hist_cols = st.columns(3)
        rfm = st.session_state.rfm_data

        # Robust way to calculate stats
        stats = {}
        rfm_log = rfm.copy() # Create copy for log transforms
        # Calculate log transforms, handle potential zeros/negatives if necessary
        rfm_log['Frequency_log'] = np.log1p(rfm['Frequency'])
        rfm_log['Monetary_log'] = np.log1p(rfm[rfm['Monetary'] > 0]['Monetary']) # Log only positive values
        
        for col in ['Recency', 'Frequency', 'Monetary']:
            try:
                # Calculate stats on ORIGINAL data for lines
                stats[col] = {
                    'mean': rfm[col].mean(),
                    'median': rfm[col].median(),
                    'q1': rfm[col].quantile(0.25),
                    'q3': rfm[col].quantile(0.75)
                }
            except Exception as e:
                 st.warning(f"Could not calculate stats for {col}: {e}")
                 stats[col] = None # Mark as None if calculation fails

        def add_dist_lines(fig, metric_name, metric_stats, is_log_x=False): # Added is_log_x flag
            if metric_stats is None: return 
            
            median_val = metric_stats['median']
            q1_val = metric_stats['q1']
            q3_val = metric_stats['q3']
            
            # Format annotation text 
            median_text = f"Median: {median_val:.1f}" if metric_name != 'Monetary' else f"Median: {format_currency(median_val)}"
            
            # Add Median Line 
            fig.add_vline(x=median_val, line_width=2, line_dash="dash", line_color="#555555", 
                          annotation_text=median_text, annotation_position="top left",
                          annotation_font_color="#555555")
            # Add Quantile Lines 
            # Only add if they are positive for log scale
            if not is_log_x or q1_val > 0:
                 fig.add_vline(x=q1_val, line_width=1, line_dash="dot", line_color=COLORS["accent"])
            if not is_log_x or q3_val > 0:
                 fig.add_vline(x=q3_val, line_width=1, line_dash="dot", line_color=COLORS["accent"],
                                annotation_text="P25-P75", annotation_position="bottom right", 
                                annotation_font_color=COLORS["accent"])

        # Plotting logic using the robust stats
        with hist_cols[0]:
            st.markdown("**Recency (Days Ago)**")
            if stats['Recency']:
                try:
                    fig_r = px.histogram(rfm, x="Recency", nbins=40) # More bins
                    base_layout_r = PLOTLY_TEMPLATE["layout"].copy()
                    base_layout_r.pop('title', None)
                    fig_r.update_layout(**base_layout_r, bargap=0.1, yaxis_title="Customer Count")
                    add_dist_lines(fig_r, 'Recency', stats['Recency'], is_log_x=False)
                    st.plotly_chart(fig_r, use_container_width=True)
                    st.caption("⬅️ Most customers are recent (low days). A long tail indicates some lapsed customers.")
                except Exception as e:
                    st.error(f"Hist Error (R): {e}")
            else:
                st.info("Could not display Recency histogram.")
        with hist_cols[1]:
            st.markdown("**Frequency (Total Orders)**")
            if stats['Frequency']:
                try:
                    # Plot LOG TRANSFORMED data on a LINEAR axis
                    fig_f = px.histogram(rfm_log, x="Frequency_log", nbins=40)
                    base_layout_f = PLOTLY_TEMPLATE["layout"].copy()
                    base_layout_f.pop('title', None)
                    fig_f.update_layout(**base_layout_f, bargap=0.1, yaxis_title="Customer Count", 
                                      xaxis_title="Number of Orders (Log Scale)")
                    # Add lines using ORIGINAL stats (Plotly places them correctly on linear axis)
                    # Note: Lines might be less meaningful if distribution is heavily transformed
                    # add_dist_lines(fig_f, 'Frequency', stats['Frequency'], is_log_x=False) # Keep is_log_x=False as axis is linear now
                    st.plotly_chart(fig_f, use_container_width=True)
                    st.caption("➡️ Distribution shown on a log scale. Most order infrequently; tail shows frequent buyers.")
                except Exception as e:
                    st.error(f"Hist Error (F): {e}")
            else:
                 st.info("Could not display Frequency histogram.")
        with hist_cols[2]:
             st.markdown("**Monetary Value ($)**")
             if stats['Monetary']:
                try:
                    # Plot LOG TRANSFORMED data on a LINEAR axis
                    fig_m = px.histogram(rfm_log, x="Monetary_log", nbins=40)
                    base_layout_m = PLOTLY_TEMPLATE["layout"].copy()
                    base_layout_m.pop('title', None)
                    fig_m.update_layout(**base_layout_m, bargap=0.1, yaxis_title="Customer Count", 
                                      xaxis_title="Total Spend ($) (Log Scale)")
                    # add_dist_lines(fig_m, 'Monetary', stats['Monetary'], is_log_x=False) # Keep is_log_x=False
                    st.plotly_chart(fig_m, use_container_width=True)
                    st.caption("➡️ Distribution shown on a log scale. Most spend less; tail shows high-value customers.")
                except Exception as e:
                    st.error(f"Hist Error (M): {e}")
             else:
                 st.info("Could not display Monetary histogram.")
        
        st.write("---")
        st.subheader("🔍 Combined Patterns (Relationships between RFM)")
        st.markdown("**How do the RFM values relate to each other?**")
        col_plot1, col_plot2 = st.columns(2)
        
        with col_plot1:
            st.markdown("**Who buys frequently & recently?**")
            st.caption("Ideal customers are bottom-left (low recency, high frequency). Color shows spending (darker = more $).", unsafe_allow_html=False)
            try:
                fig1 = px.scatter(
                    st.session_state.rfm_data, x="Recency", y="Frequency", 
                    color="Monetary", color_continuous_scale=px.colors.sequential.Greys,
                    hover_name="CustomerID", hover_data={"Recency": True, "Frequency": True, "Monetary": ':$,.2f'}
                )
                fig1.update_layout(coloraxis_showscale=False, 
                                   xaxis_title="Days Since Last Purchase (Recency)",
                                   yaxis_title="Number of Orders (Frequency)")
                st.plotly_chart(fig1, use_container_width=True)
            except Exception as e:
                st.error(f"Plot Error (R vs F): {e}")

        with col_plot2:
            st.markdown("**Who buys frequently & spends more?**")
            st.caption("Ideal customers are top-right (high frequency, high spending). Color shows recency (darker = more recent).", unsafe_allow_html=False)
            try:
                fig2 = px.scatter(
                    st.session_state.rfm_data, x="Frequency", y="Monetary", 
                    color="Recency", color_continuous_scale=px.colors.sequential.Greys_r,
                    hover_name="CustomerID", hover_data={"Recency": True, "Frequency": True, "Monetary": ':$,.2f'}
                )
                fig2.update_layout(coloraxis_showscale=False, 
                                   xaxis_title="Number of Orders (Frequency)",
                                   yaxis_title="Total Spent ($)")
                st.plotly_chart(fig2, use_container_width=True)
            except Exception as e:
                st.error(f"Plot Error (F vs M): {e}")
        
        # Keep data table in expander
        with st.expander("View Raw RFM Data & Statistics"):
            st.dataframe(st.session_state.rfm_data.describe().style.format("{:.1f}"), use_container_width=True)
            st.dataframe(st.session_state.rfm_data.style.format({'Monetary': '${:,.2f}'}), use_container_width=True)
            
    else:
        st.info("RFM data not available to visualize.")

    # Navigation
    st.write("---")
    st.markdown("---")
    st.markdown("**Next Step:** Now that we understand the overall customer behavior, let's group them into distinct segments.")
    col1, col2 = st.columns(2)
    with col1:
        if st.button("⬅️ Back to Data Selection", key="back_to_sel_2"):
             st.session_state.current_page = "1. Data Selection"
             st.experimental_rerun()
    with col2:
         if st.button("➡️ Go to Customer Segmentation", key="nav_to_segment"):
            st.session_state.current_page = "3. Customer Segmentation"
            st.experimental_rerun()

def display_customer_segmentation():
    """Displays the Customer Segmentation page (Step 3) with clearer results."""
    st.header("3. 🎯 Customer Segmentation")
    st.subheader("Grouping Similar Customers")
    if not st.session_state.get('data_processed', False):
        st.warning("Data has not been processed yet. Please go back to '1. Data Selection'.")
        if st.button("Go back to Data Selection", key="back_to_sel_3"):
             st.session_state.current_page = "1. Data Selection"
             st.experimental_rerun()
        return

    st.markdown("**Goal:** Group customers with similar RFM patterns into distinct segments for targeted actions.")

    # --- Section 1: Find Optimal K (Optional) ---
    with st.expander("Analyze Optimal Number of Segments (Advanced/Optional)"): # Renamed expander
        st.markdown("Use the Elbow Method as a technical guide to find a suggested number of segments (K). You can skip this and choose K manually below.") # Simplified text
        col1_elbow, col2_elbow = st.columns([1, 3])
        
        with col1_elbow:
            # Button to trigger analysis - using a very specific key
            analyze_elbow_btn_clicked = st.button("Analyze Elbow", key="segmentation_analyze_elbow_button")
            
            # Display suggested K and button to use it (only if optimal_clusters is set)
            if st.session_state.optimal_clusters is not None:
                st.metric("Suggested K", st.session_state.optimal_clusters)
                if st.button(f"Set K = {st.session_state.optimal_clusters}", key="segmentation_set_optimal_k_button"):
                    st.session_state.n_clusters = st.session_state.optimal_clusters
                    st.success(f"K set to {st.session_state.n_clusters}. Adjust below or run segmentation.")
                    # No rerun needed, state change updates the number_input below
                    
        with col2_elbow:
            # Perform analysis ONLY if button was clicked in this run
            if analyze_elbow_btn_clicked:
                if st.session_state.normalized_data is not None:
                    with st.spinner("Calculating Elbow Method..."):
                        try:
                            X = st.session_state.normalized_data[['Recency', 'Frequency', 'Monetary']].values
                            st.session_state.k_values, st.session_state.inertia = determine_optimal_clusters(X)
                            
                            # Update optimal K in state immediately after calculation
                            new_optimal_k = None
                            if len(st.session_state.inertia) >= 3: # Need at least 3 points for 2nd derivative
                                diffs = np.diff(st.session_state.inertia, 2)
                                optimal_k_index = np.argmax(diffs) + 1 # Index in k_values[1:] due to diff
                                if optimal_k_index < len(st.session_state.k_values) -1:
                                    new_optimal_k = st.session_state.k_values[optimal_k_index + 1] # +1 because k_values starts from 1
                                else:
                                    new_optimal_k = st.session_state.k_values[-1] # Default to max K tested
                            elif len(st.session_state.k_values) > 0:
                                new_optimal_k = st.session_state.k_values[-1] # Default if too few points
                                
                            st.session_state.optimal_clusters = new_optimal_k
                            # Let Streamlit rerun naturally after state update
                        except Exception as e:
                            st.error(f"Error calculating Elbow Method: {e}")
                            st.session_state.optimal_clusters = None
                            st.session_state.k_values = None
                            st.session_state.inertia = None
                else:
                    st.warning("Normalized data not available.")
            
            # Display plot if data exists in state
            if st.session_state.k_values and st.session_state.inertia:
                 try:
                     fig = plot_elbow_method(st.session_state.k_values, st.session_state.inertia)
                     st.plotly_chart(fig, use_container_width=True)
                 except Exception as e:
                      st.error(f"Error plotting Elbow Method: {e}")
            elif analyze_elbow_btn_clicked: # Show message only if analysis was just attempted and failed data check
                 pass # Error/Warning shown above
            else:
                 st.caption("Click 'Analyze Elbow' to generate the plot.")

    st.write("---")

    # --- Section 2: Select K and Run Clustering ---
    st.subheader("Segment Your Customers")
    
    # --- Add Feature Selection Widget --- 
    st.markdown("**A. Choose Features for Clustering** (Select 1-3)")
    available_features = ['Recency', 'Frequency', 'Monetary']
    if "selected_features" not in st.session_state:
        st.session_state.selected_features = available_features # Default
        
    # Use a columns layout for feature selector and K selector
    col1_feat, col2_k = st.columns([2, 1])
    with col1_feat:
        selected_features = st.multiselect(
            "Features:",
            options=available_features,
            default=st.session_state.selected_features,
            max_selections=3,
            key="feature_selector_final",
            label_visibility="collapsed"
        )
    
    # Update state and clear results if features change
    if selected_features != st.session_state.selected_features:
         st.session_state.selected_features = selected_features
         st.session_state.cluster_labels = None
         st.session_state.cluster_profiles = None
         st.session_state.optimal_clusters = None
         st.session_state.k_values = None
         st.session_state.inertia = None
         st.experimental_rerun() 

    # Validation for feature selection
    if not 1 <= len(selected_features) <= 3:
        st.warning("Please select between 1 and 3 features.")
        can_run_clustering = False
    else:
        can_run_clustering = True
        st.caption(f"Using: {', '.join(selected_features)}")
        
    with col2_k:
        # Number input to select K
        st.markdown("**B. Choose Segments (K)**")
        n_samples = len(st.session_state.rfm_data) if st.session_state.rfm_data is not None else 0
        max_k_allowed = max(1, n_samples) # K must be at least 1
        
        # Ensure n_clusters in state is valid before displaying number_input
        if st.session_state.n_clusters > max_k_allowed:
            st.session_state.n_clusters = max_k_allowed
        elif st.session_state.n_clusters < 1:
             st.session_state.n_clusters = 1
             
        current_k = st.number_input(
            "K:", 
            min_value=1, 
            max_value=max(10, max_k_allowed), 
            value=st.session_state.n_clusters, 
            step=1,
            key="k_selector_final",
            label_visibility="collapsed",
            help=f"Number of groups (Max: {max_k_allowed})"
        )
        if current_k != st.session_state.n_clusters:
            st.session_state.n_clusters = current_k
            st.experimental_rerun() 

    # --- Run Clustering Button ---
    st.write(" ") # Spacer
    cluster_btn_key = f"run_clustering_final_{'_'.join(selected_features)}_{current_k}" # More unique key
    run_clustering_btn_clicked = st.button(f"**Segment Customers into {current_k} Groups**", key=cluster_btn_key, type="primary", use_container_width=True, disabled=(not can_run_clustering or current_k < 1))

    if run_clustering_btn_clicked:
        if st.session_state.normalized_data is not None and n_samples > 0 and can_run_clustering:
            k_to_use = current_k 
            with st.spinner(f"Running K-Means (K={k_to_use}) on: {', '.join(selected_features)}..."):
                 try:
                     X = st.session_state.normalized_data[selected_features].values
                     st.session_state.cluster_labels, st.session_state.cluster_model = perform_clustering(X, k_to_use)
                     st.session_state.cluster_profiles = create_cluster_profiles(
                         st.session_state.normalized_data, 
                         st.session_state.rfm_data,      
                         st.session_state.cluster_labels,
                         selected_features # Use selected features
                     )
                     st.session_state.features_used_for_clustering = selected_features # Store used features
                     
                     # Map customers to clusters
                     st.session_state.customer_clusters = pd.DataFrame({
                         'CustomerID': st.session_state.rfm_data['CustomerID'],
                         'Cluster': st.session_state.cluster_labels
                     })
                     st.success(f"Clustering complete! {k_to_use} customer segments created.")
                 except Exception as e:
                     st.error(f"Error running clustering: {e}")
        else:
            if not can_run_clustering:
                 st.warning("Select 1-3 features before running segmentation.")
            else:
                 st.warning("Cannot run clustering. Ensure data is loaded and normalized.")

    # --- Section 3: Display Results --- 
    if st.session_state.cluster_profiles is not None:
        # Check if profiles match current feature selection
        profile_features = [col for col in st.session_state.cluster_profiles.columns if col in available_features]
        if set(profile_features) == set(st.session_state.selected_features):
             st.write("---")
             st.subheader(f"📊 Segmentation Results ({', '.join(st.session_state.features_used_for_clustering)})" ) 
             st.markdown("**Key Customer Segments Found:**")
             st.markdown("Here are the distinct customer segments found based on their RFM behavior:")
             
             # Define action hints based on segment description
             action_hints = {
                 "⭐ VIP Customers": "💎 Nurture & Offer Exclusives",
                 "💚 Loyal Regular": "🎁 Reward Loyalty, Upsell",
                 "👋 New / Potential": "👋 Onboard & Engage",
                 "💰 Occasional Big": "🏷️ Personalize High-Value Offers",
                 "⚠️ At Risk / Churned": "🔥 Re-engagement Campaign!",
                 "👻 Lapsed/Lost": "🗑️ Win-back Offers or Archive",
                 "General Segment": "👀 Monitor / Generic Offers",
                 # Add more specific combinations if needed based on the basic terms
                 "Recent, Frequent, Low Value": "📈 Encourage Higher Cart Value",
                 "Recent, Infrequent, High Value": "🔁 Increase Purchase Frequency",
                 "Lapsed, Frequent, High Value": "❓ Investigate & Re-engage VIPs"
             }

             # Display Profiles Table with Description and Action Hint
             profiles_display = st.session_state.cluster_profiles.copy()
             if st.session_state.rfm_data is not None:
                 try:
                     profiles_display['Typical Customer'] = profiles_display.apply(get_segment_description, axis=1)
                     # Add Action Hint column
                     # Map known descriptions, then try to build one from components if no direct match
                     def get_action(desc):
                         hint = action_hints.get(desc)
                         if hint: return hint
                         # Try building hint from components
                         if "Lapsed" in desc and "Infrequent" in desc: return action_hints["⚠️ At Risk / Churned"] # Catch variations
                         if "Recent" in desc and "Frequent" in desc: return action_hints.get("Recent, Frequent", "Promote Loyalty")
                         if "Recent" in desc and "Low Value" in desc: return action_hints.get("Recent, Low Value", "Increase Order Value")
                         return action_hints["General Segment"] # Default
                     
                     profiles_display['Action Suggestion'] = profiles_display['Typical Customer'].apply(get_action)
                     
                     # Select and reorder columns for final display
                     display_cols = ['Typical Customer', 'Size', 'Percentage', 'Action Suggestion', 'Recency', 'Frequency', 'Monetary']
                     display_cols = [col for col in display_cols if col in profiles_display.columns]
                     
                     st.markdown("**Segment Profiles Overview:**") # Add title to table
                     st.dataframe(profiles_display[display_cols].style.format({
                          'Percentage': '{:.1f}%',
                          'Recency': '{:.1f} days',
                          'Frequency': '{:.1f} orders',
                          'Monetary': '${:,.2f}'
                      }).set_properties(**{
                          'text-align': 'left' 
                      }).set_table_styles([dict(selector='th', props=[('text-align', 'left')])]),
                      use_container_width=True)
                 except Exception as e:
                      st.error(f"Could not generate segment descriptions/hints: {e}")
                      st.dataframe(st.session_state.cluster_profiles) 
             else:
                 st.dataframe(st.session_state.cluster_profiles)

             # Visualizations in Expanders
             with st.expander("View Technical Cluster Visualizations (Optional)"):
                 # Cluster 3D Plot (Conditional)
                 st.markdown("**Segments in Feature Space**")
                 if len(st.session_state.features_used_for_clustering) == 3:
                     if st.session_state.rfm_data is not None and st.session_state.cluster_labels is not None:
                          try:
                              # Assuming plot_rfm_3d primarily uses rfm_data and labels, 
                              # but ideally it should also accept the feature names to plot.
                              # For now, let's assume it plots R,F,M if 3 features were used for clustering.
                              # A more robust solution would adapt plot_rfm_3d. 
                              if set(st.session_state.features_used_for_clustering) == {'Recency', 'Frequency', 'Monetary'}:
                                   fig_3d = plot_rfm_3d(st.session_state.rfm_data, st.session_state.cluster_labels)
                                   st.plotly_chart(fig_3d, use_container_width=True)
                              else:
                                   st.info("3D plot is shown for standard RFM features only.")
                          except Exception as e:
                              st.error(f"Could not generate Cluster 3D plot: {e}")
                     else:
                          st.caption("Data or cluster labels missing for 3D plot.")
                 elif len(st.session_state.features_used_for_clustering) == 2:
                      st.info("2D scatter plot visualization for 2 features could be added here.")
                 else: 
                      st.info("3D visualization requires exactly 3 features to be selected for clustering.")
                 
                 st.write("---")
                 # Individual Cluster Radar Plots (Should adapt automatically if plot_cluster_profiles is robust)
                 st.markdown("**Individual Segment Profiles (Radar Chart)**")
                 st.caption(f"Shows how each segment scores on {', '.join(st.session_state.features_used_for_clustering)} (normalized 0-1 based on range of averages)." )
                 try:
                     # Pass the current profiles which should be based on selected features
                     radar_charts = plot_cluster_profiles(st.session_state.cluster_profiles) 
                     if radar_charts: # Check if list is not empty
                          num_charts = len(radar_charts)
                          # Display charts in columns, max 4 per row
                          num_cols = min(num_charts, 4) 
                          cols = st.columns(num_cols)
                          for i, chart in enumerate(radar_charts):
                              with cols[i % num_cols]:
                                  st.plotly_chart(chart, use_container_width=True)
                     else:
                          st.info("No radar charts generated.") # Handle empty list case
                 except Exception as e:
                      st.error(f"Could not generate Radar plot(s): {e}")
        else:
            st.info("Segmentation results shown below are based on previous feature selection. Re-run segmentation with current features if needed.")
            # Optionally display the old results anyway or clear them
            # st.dataframe(st.session_state.cluster_profiles) 

    # --- Navigation --- 
    st.markdown("--- ") # Final separator
    st.markdown("**Next Step:** Now that we have segments, let's see what products to recommend to each group.") # Transition text
    col1_nav, col2_nav = st.columns(2)
    with col1_nav:
        if st.button("⬅️ Back to Data Exploration", key="back_to_explore_final"):
             st.session_state.current_page = "2. Data Exploration"
             st.experimental_rerun()
    with col2_nav:
        # Enable button only if clustering is done
        clustering_done = st.session_state.cluster_labels is not None
        if st.button("➡️ Go to Recommendations", key="nav_to_recs_final", disabled=not clustering_done, type="primary"):
            st.session_state.current_page = "4. Recommendations"
            st.experimental_rerun()

def display_recommendations():
    """Displays the Recommendations page (Step 4) with improved clarity."""
    st.header("4. 🛍️ Product Recommendations") 
    st.subheader("Suggesting Relevant Products by Segment")
    # --- Prerequisite Checks ---
    if not st.session_state.get('data_processed', False):
        st.warning("Data has not been processed yet. Go back to Step 1.")
        if st.button("Go to Data Selection", key="recs_back_to_sel_1"):
            st.session_state.current_page = "1. Data Selection"
            st.experimental_rerun()
        return
    if st.session_state.get('customer_clusters') is None:
         st.warning("Customer segmentation has not been performed yet. Go back to Step 3.")
         if st.button("Go to Segmentation", key="recs_back_to_seg_1"):
            st.session_state.current_page = "3. Customer Segmentation"
            st.experimental_rerun()
         return

    st.markdown("**Goal:** Identify products likely to appeal to each customer segment to personalize marketing.")
    st.info("💡 **Action Idea:** Use these lists for targeted email campaigns, website personalization, or promotions.")
    
    st.write("---")
    st.subheader("Recommendations by Segment")
    if st.session_state.cluster_profiles is not None:
        segment_list = sorted(st.session_state.cluster_profiles.index.tolist())
        
        def format_segment_option(segment_id):
            try: 
                profile = st.session_state.cluster_profiles.loc[segment_id]
                desc = get_segment_description(profile) 
                return f"Segment {segment_id} - {desc}"
            except Exception:
                 return f"Segment {segment_id}"
            
        selected_segment = st.selectbox(
            "Select a customer segment:", 
            segment_list, 
            format_func=format_segment_option, 
            key="segment_selector_recs_final_v2" # New key
        )
         
        if selected_segment is not None: 
             recs_key = f"recs_for_segment_{selected_segment}"
             if recs_key not in st.session_state:
                 with st.spinner(f"Generating recommendations for Segment {selected_segment}..."):
                     try:
                         st.session_state[recs_key] = get_cluster_recommendations(
                             st.session_state.cleaned_data,
                             st.session_state.customer_clusters,
                             selected_segment
                         )
                     except Exception as e:
                         st.error(f"Error generating recommendations: {e}")
                         st.session_state[recs_key] = pd.DataFrame() # Assign empty df on error
                     
             recs = st.session_state.get(recs_key)
             
             # Get segment description again for the title
             segment_desc = "" 
             try: 
                 segment_desc = get_segment_description(st.session_state.cluster_profiles.loc[selected_segment])
             except: pass
             
             st.markdown(f"**Top Product Suggestions for:** `{segment_desc}` (Segment {selected_segment})") # Enhanced title
             if recs is None:
                 st.warning(f"Could not generate recommendations.")
             elif recs.empty:
                 st.info(f"No specific product recommendations found for this segment.")
             else:
                 # Display top recommendations using columns 
                 top_n = 5
                 num_cols = min(top_n, 5) # Allow up to 5 columns for recs
                 cols = st.columns(num_cols)
                 for i, (_, row) in enumerate(recs.head(top_n).iterrows()):
                     with cols[i % num_cols]:
                         # Simple display: Name and Price/Metric
                         st.markdown(f"**{row['Description']}**")
                         st.caption(f"ID: {row['StockCode']}")
                         st.metric(label="Avg Price", value=f"{format_currency(row['UnitPrice'])}", delta=None)
                         st.metric(label="Units in Seg.", value=f"{int(row['TotalQuantity'])}", delta=None)
                         if i < num_cols - 1:
                              st.write(" ") # Add space below metrics unless last column

                 # Optional: Expander for full table and plot
                 with st.expander(f"View Full List & Revenue Plot (Top {min(10, len(recs))})"): 
                     st.dataframe(recs.head(10)[["Description", "TotalQuantity", "UnitPrice", "TotalRevenue"]].style.format({'UnitPrice': '${:,.2f}', 'TotalRevenue': '${:,.2f}'}), use_container_width=True)
                     try:
                         fig = plot_recommendations(recs.head(10))
                         st.plotly_chart(fig, use_container_width=True)
                     except Exception as plot_e:
                         st.error(f"Could not plot recommendations: {plot_e}")
    else:
         st.info("Segment profiles not available. Run segmentation (Step 3) first.")

    # --- Navigation --- 
    st.write("---")
    col1, col2 = st.columns(2)
    with col1:
        if st.button("⬅️ Back to Segmentation", key="recs_back_to_segment"):
             st.session_state.current_page = "3. Customer Segmentation"
             st.experimental_rerun()
    with col2:
        if st.button("➡️ Go to Interactive Demo", key="recs_nav_to_demo", type="primary"):
            st.session_state.current_page = "5. Interactive Demo"
            st.experimental_rerun()

def display_interactive_demo():
    """Displays the Interactive Demo page (Step 5) with clearer layout."""
    st.header("5. 🎮 Interactive Demo: What-If Scenarios") 
    st.subheader("Simulate Profiles & See Outcomes")
    # --- Prerequisite Checks ---
    if not st.session_state.get('data_processed', False):
        st.warning("Data has not been processed yet. Go back to Step 1.")
        # Add back button
        return
    # Check if clustering used the standard RFM features needed for this demo
    standard_features = ['Recency', 'Frequency', 'Monetary']
    clustering_features = st.session_state.get("features_used_for_clustering")
    if clustering_features is None:
         st.warning("Segmentation not performed yet (Step 3).")
         return
    elif set(clustering_features) != set(standard_features):
         st.info(f"Interactive demo requires segmentation based on Recency, Frequency, and Monetary features. Current segmentation used: {clustering_features}. Please re-run Step 3 with RFM.")
         return
    # Check for model and scaler (safe check)
    if st.session_state.get('cluster_model') is None or st.session_state.get('scaler') is None:
         st.warning("Clustering model/scaler not available. Please re-run segmentation (Step 3)." )
         return
         
    st.markdown("**Goal:** Experiment with different customer profiles (RFM values) to understand how they map to segments and influence recommendations.")
    st.info("💡 **Use Case:** See how improving Recency or Frequency might shift a customer to a more valuable segment.")

    # --- Input Sliders ---
    st.subheader("Build a Customer Profile")
    col_r, col_f, col_m = st.columns(3)
    # Use min/max from actual data if available, otherwise use sensible defaults
    r_max = int(st.session_state.rfm_data['Recency'].max()) if st.session_state.rfm_data is not None else 365
    f_max = int(st.session_state.rfm_data['Frequency'].max()) if st.session_state.rfm_data is not None else 50
    m_max = int(st.session_state.rfm_data['Monetary'].max()) if st.session_state.rfm_data is not None else 5000
    
    with col_r:
        rec = st.slider("Recency (Days Ago)", 0, r_max, 50, key="demo_rec_slider")
    with col_f:
        freq = st.slider("Frequency (Total Orders)", 1, f_max, 5, key="demo_freq_slider")
    with col_m:
        mon = st.slider("Monetary Value ($)", 0, m_max, 200, key="demo_mon_slider")

    # --- Prediction and Recommendations --- 
    st.write("---")
    predict_col1, predict_col2 = st.columns([1, 3]) 
    with predict_col1:
        run_prediction = st.button("**Analyze This Profile**", key="demo_predict_final_btn_v3", type="primary", use_container_width=True) # New key

    result_placeholder = st.empty()

    if run_prediction:
        # Clear previous demo results before running new prediction
        if "demo_predicted_cluster" in st.session_state: del st.session_state.demo_predicted_cluster
        if "demo_recommendations" in st.session_state: del st.session_state.demo_recommendations
        try:
            with st.spinner("Predicting segment and getting recommendations..."):
                # Scale input
                input_data = np.array([[rec, freq, mon]])
                input_scaled = st.session_state.scaler.transform(input_data)
                
                # Predict cluster
                predicted_cluster = st.session_state.cluster_model.predict(input_scaled)[0]
                st.session_state.demo_predicted_cluster = predicted_cluster # Store in state

                # Get recommendations for that cluster
                recs = get_cluster_recommendations(
                    st.session_state.cleaned_data,
                    st.session_state.customer_clusters,
                    predicted_cluster
                )
                st.session_state.demo_recommendations = recs # Store in state
                
                # Force rerun to display results in the placeholder below
                st.experimental_rerun() 

        except Exception as e:
            result_placeholder.error(f"Error during prediction/recommendation: {e}")
            st.session_state.demo_predicted_cluster = None
            st.session_state.demo_recommendations = None
                 
    # --- Display Demo Results (if they exist in state) ---
    if "demo_predicted_cluster" in st.session_state and st.session_state.demo_predicted_cluster is not None:
        with result_placeholder.container(): 
             predicted_cluster = st.session_state.demo_predicted_cluster
             
             res_col1, res_col2 = st.columns([1,2])
             
             with res_col1:
                 st.markdown("**Predicted Profile:**") # Changed title
                 st.metric(label="Segment ID", value=predicted_cluster)
                 # Show segment description more prominently
                 if st.session_state.cluster_profiles is not None:
                     try:
                         profile = st.session_state.cluster_profiles.loc[predicted_cluster]
                         desc = get_segment_description(profile) 
                         st.info(f"**Type:** {desc}" ) # Display description clearly in an info box
                     except KeyError: 
                         st.warning(f"Profile for segment {predicted_cluster} not found.")
                     except Exception as e:
                         st.error(f"Error getting segment description: {e}")
             
             with res_col2:
                 st.markdown("**Example Recommendations:**") # Changed title
                 if "demo_recommendations" in st.session_state and st.session_state.demo_recommendations is not None:
                     recs = st.session_state.demo_recommendations
                     if recs.empty:
                         st.info(f"No specific recommendations found for this profile's segment.")
                     else:
                         # Display simpler list with bullets for demo
                         for _, row in recs.head(5).iterrows():
                            st.markdown(f"* {row['Description']} ({format_currency(row['UnitPrice'])})")
                         # Removed expander for demo simplicity
                 else:
                     st.warning("Recommendations could not be generated.")

    # --- Navigation --- 
    st.write("---")
    if st.button("⬅️ Back to Recommendations", key="demo_back_to_recs_final_v2"):
         st.session_state.current_page = "4. Recommendations"
         # Clear demo state when navigating away
         if "demo_predicted_cluster" in st.session_state: del st.session_state.demo_predicted_cluster
         if "demo_recommendations" in st.session_state: del st.session_state.demo_recommendations
         st.experimental_rerun()


# --- Main Application Logic ---
def main_app():
    """Main function to run the Streamlit application."""
    # Basic setup
    setup_page("E-commerce Recommendation System Refactored", "🛍️")
    
    # Initialize state ONCE
    initialize_app_state()

    # --- Sidebar Navigation ---
    st.sidebar.title("Workflow")
    # Use index based on WORKFLOW_STEPS list
    current_page_index = WORKFLOW_STEPS.index(st.session_state.current_page)
    
    selected_page = st.sidebar.radio(
        "Go to Step:", 
        WORKFLOW_STEPS, 
        index=current_page_index,
        key="sidebar_nav"
    )

    # Update page if selection changes via sidebar
    if selected_page != st.session_state.current_page:
        st.session_state.current_page = selected_page
        st.experimental_rerun() # Rerun to display the new page

    # --- Page Content Display ---
    # Simple header
    st.title("E-commerce Recommendation System")
    st.markdown("*Customer Segmentation & Product Recommendations*")
    st.write("---")

    # Display the selected page content using a dictionary lookup
    page_functions = {
        "1. Data Selection": display_home_page,
        "2. Data Exploration": display_data_exploration,
        "3. Customer Segmentation": display_customer_segmentation,
        "4. Recommendations": display_recommendations,
        "5. Interactive Demo": display_interactive_demo,
    }
    
    # Get the function corresponding to the current page and call it
    render_function = page_functions.get(st.session_state.current_page)
    if render_function:
        render_function()
    else:
        st.error("Page not found. Please select a valid step from the sidebar.")

# Run the application
if __name__ == "__main__":
    main_app() 