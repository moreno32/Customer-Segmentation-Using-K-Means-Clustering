# E-commerce Recommendation System - Project Structure

## Core Project Structure

```
ecommerce-recommendation-system/ # Root directory
│
├── app/                     # Source code for the Streamlit application
│   ├── main.py              # Main script: UI, workflow control, page routing
│   ├── data.py              # Data operations: loading (URL, upload, synthetic), cleaning, RFM calculation
│   ├── clustering.py        # Clustering logic: K-Means, Elbow Method, profile calculation
│   ├── recommender.py       # Recommendation algorithms (segment-based)
│   ├── visualization.py     # Plotly functions for creating visualizations
│   ├── utils.py             # Utility functions: page setup, formatting, etc.
│   └── styles.py            # Styling definitions: COLORS dict, PLOTLY_TEMPLATE
│
├── data/                    # Default location for downloaded/cached data (if applicable)
│
├── docs/                    # Project documentation files
│   ├── README.md            # Documentation specific to the docs folder (optional)
│   ├── application_flow.md  # Detailed description of the app's step-by-step logic
│   ├── project_documentation.md # Overall project goals, tech stack, features (this file)
│   └── project_structure.md # Description of the file/folder layout
│
├── tests/                   # Directory for unit/integration tests (if implemented)
│   ├── test_data.py         # Example: Tests for data module
│   └── ...                  # Other test files
│
├── requirements.txt         # List of Python dependencies for pip
├── run_app.py               # Original script to run the app (use `streamlit run` instead if issues)
├── run_tests.py             # Script to execute tests (if applicable)
├── .gitignore               # Specifies intentionally untracked files that Git should ignore
├── LICENSE                  # Project's software license
└── README.md                # Main project README: Overview, quick start, usage
```

## Module/File Descriptions (`app/` directory)

-   **`main.py`**: 
    -   Entry point for the Streamlit application.
    -   Handles page routing using `st.session_state`.
    -   Defines the UI layout for each step (`display_...` functions).
    -   Orchestrates calls to other modules for data processing, clustering, etc.
-   **`data.py`**: 
    -   `load_data`: Handles loading from URL/upload, includes robust synthetic data fallback.
    -   `create_sample_dataset`: Generates realistic synthetic data based on customer archetypes.
    -   `clean_data`: Performs necessary data cleaning steps.
    -   `create_rfm_features`: Calculates Recency, Frequency, Monetary value.
    -   `normalize_data`: Applies StandardScaler to selected features.
-   **`clustering.py`**: 
    -   `determine_optimal_clusters`: Implements Elbow Method (adapts K based on sample size).
    -   `perform_clustering`: Executes K-Means algorithm.
    -   `create_cluster_profiles`: Calculates average feature values for each cluster.
-   **`recommender.py`**: 
    -   `get_cluster_recommendations`: Generates product recommendations based on segment purchase history (e.g., top N products by revenue/quantity within segment).
    -   Potentially other recommendation logic (e.g., `get_customer_recommendations` if implemented).
-   **`visualization.py`**: 
    -   Defines `PLOTLY_TEMPLATE` for consistent styling.
    -   Contains functions (`plot_rfm_3d`, `plot_elbow_method`, `plot_cluster_profiles`, `plot_recommendations`, etc.) to generate specific Plotly figures.
    -   Ensures monochromatic color scheme and clear labeling.
-   **`utils.py`**: 
    -   `setup_page`: Configures Streamlit page settings.
    -   `format_currency`, `format_number`: Basic formatting helpers.
    -   Potentially other shared utility functions.
-   **`styles.py`**: 
    -   Defines the `COLORS` dictionary.
    -   Defines the `PLOTLY_TEMPLATE` dictionary for consistent Plotly chart styling.
    -   May contain other styling constants or functions if needed.

## Key Design Choices Reflected in Structure

-   **Modularity:** Each core functionality (data, clustering, recommending, visualizing) resides in its own module.
-   **Separation of Concerns:** `main.py` focuses on UI and flow control, while other modules handle specific tasks.
-   **Configuration/Styling Centralized:** `styles.py` and potentially parts of `utils.py` centralize visual aspects.
-   **Robust Data Loading:** `data.py` encapsulates the logic for handling real vs. synthetic data.
-   **Ad-hoc Capability:** Logic for handling user-selected features resides primarily within `main.py` (UI) and the functions in `clustering.py` and `visualization.py` are designed to accept dynamic feature lists. 