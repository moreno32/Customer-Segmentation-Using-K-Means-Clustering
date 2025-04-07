# E-commerce Personalized Recommendation System
## Project Documentation

## 1. Project Overview
This project provides a streamlined and interactive application for **customer segmentation** and **product recommendation** in an e-commerce context. Built with Streamlit, it demonstrates a typical data science workflow, starting from data loading and exploration (using RFM analysis) to customer clustering (K-Means) and finally generating targeted recommendations. The application features a clean, professional, monochromatic interface focused on clarity and stakeholder understanding.

## 2. Key Objectives
- Demonstrate a practical workflow for RFM-based customer segmentation.
- Provide clear, interpretable visualizations of customer behavior and segments.
- Implement K-means clustering with user-selectable features (Recency, Frequency, Monetary).
- Generate actionable product recommendations based on identified customer segments.
- Offer an interactive demo for exploring "what-if" scenarios.
- Maintain a simple, intuitive, and professional user interface.

## 3. Design Philosophy
- **Clarity First:** Prioritize clear explanations, intuitive interfaces, and easily understandable visualizations.
- **Stakeholder Value:** Emphasize business insights (e.g., segment descriptions, action suggestions) and the practical application of results.
- **Minimalist & Professional:** Utilize a clean, monochromatic (grayscale) design with ample white space and clear typography.
- **Robust & Simple:** Favor native Streamlit components, avoid overly complex code structures, ensure reliable navigation and data handling (including synthetic data fallback).

## 4. Application Workflow (Steps in `app/main.py`)

1.  **Load & Prepare Data:** 
    - Select data source (Example or Upload CSV).
    - Validate uploaded CSV immediately (required columns: `CustomerID`, `InvoiceDate`, `Quantity`, `UnitPrice`).
    - Load data (uses robust synthetic data generator as fallback).
    - Set analysis parameters (Number of Segments K, Reference Date).
    - Process Data (Clean, Calculate RFM, Normalize).
    - Navigate automatically to Exploration.
2.  **Explore Customer Behavior (RFM):**
    - Understand RFM concepts.
    - View average RFM metrics.
    - Analyze distributions with histograms (R linear, F/M log-transformed data on linear axis) including median/quantile lines.
    - Examine relationships with 2D scatter plots (R vs F, F vs M).
3.  **Customer Segmentation:**
    - **Select Features:** Choose 1-3 features (R, F, M) for clustering.
    - **Select K:** Optionally run Elbow Method (in expander) to get suggested K or set K manually.
    - **Run Segmentation:** Perform K-Means clustering using selected features and K.
    - **Analyze Results:** View segment profiles table with key metrics, business-oriented descriptions (e.g., "⭐ VIP Customers"), and actionable suggestions.
    - **Optional Viz:** View technical plots (3D RFM - if applicable, Individual Radar Charts) in an expander.
4.  **Product Recommendations:**
    - Select a customer segment (shows description).
    - View top product recommendations for that segment, presented visually (columns/metrics) and in a table (optional expander).
5.  **Interactive Demo:**
    - *Requires segmentation using R, F, M features.*
    - Adjust RFM sliders to simulate a customer profile.
    - Predict the customer's segment and view tailored recommendations.

## 5. Technology Stack
- **Core:** Python 3.8+
- **Web Framework:** Streamlit (v1.25.0 specified in `requirements.txt`)
- **Data Handling:** Pandas, NumPy
- **Machine Learning:** Scikit-learn (KMeans, StandardScaler)
- **Visualization:** Plotly Express, Plotly Graph Objects

## 6. Code Structure (`app/` directory)
- `main.py`: Main application, UI layout, workflow control.
- `data.py`: Data loading (CSV/URL/Synthetic), cleaning, RFM calculation.
- `clustering.py`: K-Means implementation, Elbow Method, profile creation.
- `recommender.py`: Logic for generating segment-based recommendations.
- `visualization.py`: Functions to create Plotly charts (Histograms, Scatter, Radar, Bar).
- `utils.py`: Helpers (page setup, formatting).
- `styles.py`: Color definitions, Plotly template.

## 7. Key Features & Improvements
- **Ad-hoc Feature Selection:** Users can choose which RFM metrics drive segmentation.
- **Robust Data Handling:** Includes validation, synthetic data generation, and automatic reference date adjustment for synthetic data.
- **Stakeholder-Friendly Outputs:** Business descriptions for segments, actionable suggestions, clear explanations.
- **Improved Visualizations:** Log-transformed histograms, interpretable radar charts, cleaner presentation.
- **Reliable Navigation:** Refactored workflow logic using `st.session_state` and `st.experimental_rerun()`.

## 8. Running the Application
```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Run the app (recommended)
streamlit run app/main.py 
```

## 9. Future Considerations / Potential Extensions
- More sophisticated recommendation algorithms (e.g., collaborative filtering if user-item data is available).
- Option to save/load segmentation models and results.
- More advanced visualization options (e.g., interactive segment filtering).
- Integration with a database backend.
- More comprehensive testing coverage. 