# Personalized E-commerce Recommendation System

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![Streamlit](https://img.shields.io/badge/streamlit-1.25.0-red.svg)](https://streamlit.io/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.3.0-orange.svg)](https://scikit-learn.org/)

A Streamlit application demonstrating customer segmentation using RFM analysis and K-means clustering, leading to targeted product recommendations.

## Features

- **RFM Analysis**: Calculates Recency, Frequency, and Monetary value for customers.
- **Data Loading**: Supports example data and CSV uploads with validation.
- **Synthetic Data Fallback**: Generates realistic synthetic data if downloads fail or data is too small.
- **Data Exploration**: Visualizes customer behavior with:
    - Average RFM metrics.
    - Histograms for individual RFM distributions (including log-transformed views).
    - 2D Scatter plots showing relationships between RFM variables.
- **Ad-hoc Customer Segmentation**:
    - Uses K-means clustering.
    - Allows users to **select 1-3 features** (Recency, Frequency, Monetary) for clustering.
    - Optional Elbow Method analysis to suggest an optimal K.
    - Clear display of segment profiles with business-oriented descriptions (e.g., "⭐ VIP Customers", "⚠️ At Risk").
    - Technical visualizations (3D Plot - conditional, Individual Radar Charts) available in an expander.
- **Targeted Product Recommendations**: 
    - Generates recommendations based on segment purchase history.
    - Presents top recommendations visually.
- **Interactive Demo**: Simulates customer profiles (RFM) to predict segments and show corresponding recommendations (requires segmentation with R, F, M).
- **Stakeholder-Focused Interface**: Clean design, clear explanations, and actionable insights emphasized throughout the workflow.

## Quick Start

1.  **Clone:** `git clone <your-repo-url>`
2.  **Navigate:** `cd <repo-folder>`
3.  **Install:** `pip install -r requirements.txt`
4.  **Run:** `streamlit run app/main.py` (Recommended over `python run_app.py` if issues arise)
5.  **Access:** Open browser to `http://localhost:8501` (or the URL provided).

## Using the Application

The application workflow guides you through these steps:

1.  **Load & Prepare Data**: Choose example data or upload a validated CSV. Set parameters (K, Reference Date) and confirm.
2.  **Exploring Customer Behavior**: Understand RFM concepts. View average metrics, distribution histograms, and relationship scatter plots.
3.  **Customer Segmentation**: Select features (R, F, M, or a subset), optionally analyze K, choose K, and run segmentation. View segment profiles and descriptions.
4.  **Product Recommendations**: Select a generated segment to see tailored product suggestions.
5.  **Interactive Demo**: Adjust RFM sliders to simulate a customer, predict their segment, and see relevant recommendations (only if RFM were used for segmentation).

## Dataset

- Uses the [Online Retail Dataset](https://archive.ics.uci.edu/ml/datasets/online+retail) by default.
- Includes a robust synthetic data generator as a fallback and for demonstration.
- Supports uploading custom CSV files with `CustomerID`, `InvoiceDate`, `Quantity`, `UnitPrice` columns.

## Project Structure 

(Structure remains largely the same as previously documented, main files are within `app/`)

```
ecommerce-recommendation-system/
├── app/               # Main application code
│   ├── main.py        # Main Streamlit application logic & UI
│   ├── data.py        # Data loading, cleaning, synthetic generation, RFM calculation
│   ├── clustering.py  # K-Means, Elbow method, Profile creation
│   ├── recommender.py # Recommendation logic
│   ├── visualization.py # Plotly visualization functions
│   ├── utils.py       # Helper functions (page setup, formatting)
│   └── styles.py      # Styling constants (e.g., COLORS)
├── data/              # Default directory for downloaded data
├── docs/              # Project documentation
├── tests/             # Unit/Integration tests (if applicable)
├── run_app.py         # Original application runner (may have environment issues)
├── requirements.txt   # Project dependencies
└── README.md          # This file
```

## Technical Details

- **Core Libraries**: Streamlit (v1.25.0 specified in requirements), Pandas, Numpy, Scikit-learn, Plotly.
- **Methodology**: RFM Analysis, K-Means Clustering, Basic Recommendation Logic (e.g., top products per segment).

## Running Tests

(Assuming tests exist and `run_tests.py` is configured)
```bash
python run_tests.py 
```
Or directly:
```bash
python -m pytest tests/
```

## License

MIT License - see the LICENSE file.

## Acknowledgments

- Dataset provided by UCI Machine Learning Repository
- Built with Streamlit, scikit-learn, and Plotly
