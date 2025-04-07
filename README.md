# E-commerce Customer Segmentation & Recommendation App

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![Streamlit](https://img.shields.io/badge/streamlit-1.25.0-red.svg)](https://streamlit.io/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.3.0-orange.svg)](https://scikit-learn.org/)

**Unlock targeted marketing and boost sales by understanding your customers better!** This interactive Streamlit application performs **RFM analysis** and **K-Means clustering** to automatically group your customers into meaningful segments (like VIPs, At Risk, New) and suggests relevant products for each group.

--- 

## ✨ Live Demo

**[➡️ Try the Interactive App Here!](https://your-app-name.streamlit.app/)**  *(Replace with your Streamlit Cloud URL)*

--- 

## Key Features & Benefits

*   **Understand Your Customers:** Automatically analyze purchase history using **Recency, Frequency, and Monetary (RFM)** metrics.
*   **Discover Segments:** Group similar customers using **K-Means Clustering** to identify distinct behavioral patterns.
*   **Flexible Analysis:** Choose **which features** (R, F, M) drive the segmentation (1-3 features).
*   **Actionable Insights:** Get clear segment profiles with business descriptions (e.g., `⭐ VIP`, `⚠️ At Risk`) and **action suggestions**.
*   **Targeted Recommendations:** View **product recommendations** tailored to each specific customer segment.
*   **Easy Visualization:** Understand data distributions and segment differences through **clear, interactive charts** (Histograms, Scatter Plots, Radar Charts).
*   **"What-If" Scenarios:** Use the **Interactive Demo** to see how hypothetical customer profiles are segmented and what they might be recommended.
*   **Data Flexibility:** Use the built-in realistic **synthetic dataset** or easily **upload your own CSV**.

## Workflow Overview

1.  **Load Data:** Upload your CSV or use the example/synthetic data.
2.  **Explore Behavior:** Analyze RFM distributions and patterns.
3.  **Segment Customers:** Choose features & K, run clustering, view profiles.
4.  **Get Recommendations:** See suggested products for chosen segments.
5.  **Simulate Profiles:** Interactively test RFM values in the demo.

## Quick Start (Local)

1.  **Clone:** `git clone <your-repo-url>`
2.  **Install:** `cd <repo-folder> && pip install -r requirements.txt`
3.  **Run:** `streamlit run app/main.py`
4.  **Access:** Open `http://localhost:8501` in your browser.

## Tech Stack

*   **Core:** Python, Streamlit, Pandas, Scikit-learn, Plotly

## Documentation & License

*   Detailed documentation can be found in the `/docs` folder.
*   Licensed under the MIT License (see `LICENSE` file).

--- 

*(Optional: Add a better screenshot of the actual running application here)*

![Placeholder Screenshot](https://www.datascience-pm.com/wp-content/uploads/2018/09/cluster-1.png)
