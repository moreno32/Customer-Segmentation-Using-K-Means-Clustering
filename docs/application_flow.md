# Updated E-commerce Recommendation System Application Flow

## System Overview

This document outlines the step-by-step flow of the refactored e-commerce recommendation system, focusing on clarity, stakeholder value, and a simplified, robust user experience.

## Design Philosophy

- **Clarity First:** Prioritize clear explanations and intuitive interfaces.
- **Stakeholder Value:** Emphasize business insights and actionable results.
- **Minimalist & Professional:** Clean, monochromatic design (grayscale).
- **Robust & Simple:** Prefer native Streamlit components, avoid overly complex logic.

## Color Palette (Defined in `app/styles.py`)

```python
COLORS = {
    'background': '#FFFFFF',     # Pure white background
    'text': '#212121',           # Near-black for text
    'accent': '#616161',         # Medium gray for accents
    'light_accent': '#F5F5F5',   # Very light gray for cards/dividers
    'highlight': '#000000',      # Pure black for highlights
    'error': '#B00020',          # Dark red
    'success': '#2E7D32',        # Dark green 
}
PLOTLY_TEMPLATE = { ... } # Defines Plotly layout defaults (grayscale, fonts)
```

## Application Flow (`app/main.py`)

### 1. Initialization & Setup (`main_app`, `initialize_app_state`)

1.  **Launch:** User runs `streamlit run app/main.py`.
2.  **Setup:** `setup_page` configures page layout (wide) and title.
3.  **State Init:** `initialize_app_state` sets default values for session state variables (`current_page`, `data_processed`, `n_clusters`, etc.) if they don't exist.
4.  **Sidebar Nav:** `st.sidebar.radio` displays workflow steps ("1. Data Selection", "2. Data Exploration", etc.) and updates `st.session_state.current_page` on selection, triggering `st.experimental_rerun`.
5.  **Page Routing:** Based on `st.session_state.current_page`, the corresponding `display_...()` function is called to render the content.

### 2. Step 1: Load & Prepare Data (`display_home_page`)

1.  **Goal:** Load customer purchase history to begin analysis.
2.  **Data Source Selection:** `st.radio` allows choosing "Use Example Data" or "Upload CSV File".
3.  **Loading & Validation (No Form):**
    *   **Upload:** `st.file_uploader` appears. When a file is uploaded, it's **immediately read and validated** (checks for required columns: CustomerID, InvoiceDate, Quantity, UnitPrice). Success/error messages and a preview (`st.expander`) are shown instantly. Valid data is stored in `st.session_state.user_uploaded_data`.
    *   **Example:** `st.button` triggers loading (`load_data()`). If successful, data goes into `st.session_state.example_data_loaded`, preview shown. Synthetic data fallback (`create_sample_dataset` in `data.py`) is used if downloads fail or the dataset is too small (<10 unique customers after cleaning).
4.  **Parameter Setup:** `st.number_input` for "Number of Customer Groups (K)" and `st.date_input` for "Reference Date" are always visible. Values update `st.session_state.n_clusters` and `st.session_state.ref_date`.
5.  **Start Analysis Button:** A primary button `st.button("📊 Start RFM Calculation & Segmentation Prep")` is enabled **only when valid data is loaded** (either example or uploaded).
6.  **Processing & Navigation:** Clicking the button:
    *   Copies the correct data (example or uploaded) to `st.session_state.original_data`.
    *   Determines the correct Reference Date (auto for synthetic, UI for real).
    *   Runs `clean_data`, `create_rfm_features`, `normalize_data`, storing results in session state.
    *   Sets `st.session_state.data_processed = True`.
    *   Shows success message and `st.balloons`.
    *   Waits briefly (`time.sleep`).
    *   Sets `st.session_state.current_page = "2. Data Exploration"`.
    *   Calls `st.experimental_rerun()` to navigate.

### 3. Step 2: Exploring Customer Behavior (RFM) (`display_data_exploration`)

1.  **Goal:** Analyze customer behavior using Recency, Frequency, and Monetary value (RFM).
2.  **Prerequisite Check:** Verifies if `st.session_state.data_processed` is True.
3.  **RFM Explanation:** Defines R, F, M in simple terms.
4.  **Overall Snapshot:** Displays average R, F, M using `st.metric`.
5.  **Distributions (Histograms):**
    *   Shows histograms for Recency, Frequency, and Monetary Value.
    *   Frequency and Monetary use **log-transformed data on a linear axis** to handle skewness better (via `np.log1p`). Recency uses a linear axis.
    *   Includes vertical lines indicating Median and P25-P75 range (Interquartile Range) on the original scale for reference.
    *   Clear captions interpret each distribution.
6.  **Combined Patterns (Scatter Plots):**
    *   Shows 2D scatter plots: Recency vs. Frequency (color=Monetary) and Frequency vs. Monetary (color=Recency).
    *   Captions explain how to interpret idéal customer location and color intensity.
7.  **Data Table (Optional):** `st.expander` contains RFM summary statistics and the full RFM data table.
8.  **Navigation:** Buttons to go back or proceed to Segmentation.

### 4. Step 3: Customer Segmentation (`display_customer_segmentation`)

1.  **Goal:** Group similar customers for targeted actions.
2.  **Prerequisite Check:** Verifies `data_processed`.
3.  **Feature Selection (Ad-hoc):**
    *   `st.multiselect` allows user to choose 1-3 features from ['Recency', 'Frequency', 'Monetary'] for clustering. Selection stored in `st.session_state.selected_features`.
    *   Changing features clears previous clustering results and reruns.
    *   Validation ensures 1-3 features are selected.
4.  **K Selection:**
    *   **Optional Elbow Method:** `st.expander` contains a button "Analyze Elbow". Clicking it calculates inertia for different K values (up to # samples), plots the elbow curve (`plot_elbow_method`), suggests an optimal K (`st.metric`), and shows a button "Set K = ..." to update `st.session_state.n_clusters`. (Logic carefully avoids `DuplicateWidgetID`).
    *   **Manual K Selection:** `st.number_input` always shows the current `st.session_state.n_clusters`, allowing manual adjustment. Automatically validated against the number of samples.
5.  **Run Segmentation Button:** Primary button `st.button("**Segment Customers into K Groups**")` (using selected K and features) triggers clustering.
6.  **Clustering Process:**
    *   Runs K-Means (`perform_clustering` from `clustering.py`) on the normalized data using **selected features**.
    *   Generates cluster profiles (`create_cluster_profiles` from `clustering.py`) using **selected features** and original RFM data.
    *   Stores results (`cluster_labels`, `cluster_model`, `cluster_profiles`, `customer_clusters`, `features_used_for_clustering`) in session state.
7.  **Results Display:**
    *   Shows a table (`st.dataframe`) of segment profiles (`cluster_profiles`).
    *   Includes a "Typical Customer" description (e.g., "⭐ VIP", "⚠️ At Risk") generated by `get_segment_description`.
    *   Includes an "Action Suggestion" column mapped from the description.
8.  **Visualizations (Optional):** `st.expander` contains:
    *   3D Scatter Plot (`plot_rfm_3d`): Shown **only if** the 3 standard RFM features were selected for clustering.
    *   Individual Radar Charts (`plot_cluster_profiles`): Shows one radar per segment, using the **selected features** as axes. Normalization highlights differences *between* segment profiles.
9.  **Navigation:** Buttons to go back or proceed to Recommendations (enabled only after successful clustering).

### 5. Step 4: Product Recommendations (`display_recommendations`)

1.  **Goal:** Identify products likely to appeal to each segment.
2.  **Prerequisite Checks:** Verifies `data_processed` and that clusters exist.
3.  **Segment Selection:** `st.selectbox` allows choosing a segment. The segment description (e.g., "⭐ VIP") is shown in the selector.
4.  **Recommendation Generation:** When a segment is selected, recommendations are generated/retrieved (`get_cluster_recommendations`) and stored in session state (keyed by segment) to avoid re-computation.
5.  **Display:**
    *   Shows title indicating the selected segment and its description.
    *   Displays Top 5 recommendations using `st.columns` and `st.metric` (showing Avg Price, Units Sold in Segment).
    *   `st.expander` contains the full top 10 list in a table and a bar chart visualization (`plot_recommendations`).
6.  **Navigation:** Buttons to go back or proceed to Demo.

### 6. Step 5: Interactive Demo (`display_interactive_demo`)

1.  **Goal:** Experiment with "what-if" scenarios by simulating customer profiles.
2.  **Prerequisite Checks:** Verifies `data_processed`, existence of model/scaler, and **critically, that segmentation was performed using standard R, F, M features**. If not, displays an informative message.
3.  **Profile Simulation:** `st.slider` widgets allow user to define Recency, Frequency, and Monetary values.
4.  **Analysis Button:** `st.button("**Analyze This Profile**")` triggers prediction.
5.  **Prediction & Recs:**
    *   Input values are scaled using the saved `scaler`.
    *   The saved `cluster_model` predicts the segment.
    *   `get_cluster_recommendations` fetches recommendations for the predicted segment.
    *   Results (predicted segment ID, description, recommendations) are stored temporarily in session state.
    *   `st.experimental_rerun` is used to display results below the button.
6.  **Results Display:** Shows the predicted Segment ID, its "Typical Customer" description, and a simple list of the top 5 recommended products for that segment.
7.  **Navigation:** Button to go back to Recommendations. Clears demo-specific state on navigation.

---

This flow reflects the refactored application, emphasizing clear steps, stakeholder-friendly outputs, robust navigation, and the ad-hoc feature selection capability. 