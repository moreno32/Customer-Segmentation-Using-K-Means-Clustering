"""
Data Processing Module

Functions for loading, cleaning, and feature engineering of e-commerce data.
"""

import pandas as pd
import numpy as np
from datetime import datetime
from sklearn.preprocessing import StandardScaler
import os
import urllib.request
from pathlib import Path

# Default dataset URL
DEFAULT_DATASET_URL = "https://raw.githubusercontent.com/datasciencedojo/datasets/master/Online%20Retail.csv"

def load_data(file_path=None):
    """
    Load the e-commerce dataset from a file or URL.
    
    Parameters:
    -----------
    file_path : str or file-like object, optional
        The path to the CSV file, or a file-like object.
        If None, downloads the default dataset.
    
    Returns:
    --------
    pd.DataFrame
        The loaded dataset, or None if loading failed.
    """
    df = None # Initialize df
    source = "local" # Track the source
    
    try:
        # If no file path provided, use default dataset
        if file_path is None:
            source = "download"
            # Create data directory if it doesn't exist
            data_dir = Path("data")
            data_dir.mkdir(exist_ok=True)
            
            # Check if file already exists locally
            local_file = data_dir / "online_retail.csv"
            if not local_file.exists():
                print(f"Downloading dataset from {DEFAULT_DATASET_URL}")
                try:
                    urllib.request.urlretrieve(DEFAULT_DATASET_URL, local_file)
                except Exception as e:
                    try:
                        # Fallback to alternative URL if the first one fails
                        alt_url = "https://archive.ics.uci.edu/ml/machine-learning-databases/00352/Online%20Retail.csv"
                        print(f"Error with primary URL ({e}), trying alternative: {alt_url}")
                        urllib.request.urlretrieve(alt_url, local_file)
                    except Exception as e2:
                        print(f"Error downloading from primary and alternative URLs ({e2}).")
                        # --- Fallback to Synthetic Data Generation --- 
                        print("Generating larger synthetic dataset as fallback...")
                        df = create_sample_dataset() 
                        source = "synthetic"
                        # Optionally save the synthetic data locally for future runs?
                        # df.to_csv(local_file, index=False) 
                        
            # If df is still None, it means download might have succeeded or synthetic was created
            if df is None: 
                 file_path = local_file
            
        # Load data if not generated synthetically
        if source != "synthetic":
            print(f"Loading data from {file_path}...")
            df = pd.read_csv(file_path, encoding='ISO-8859-1', parse_dates=['InvoiceDate'])
            print(f"Loaded dataset with {len(df)} rows and {df.shape[1]} columns from {source}.")
            
            # --- Check if loaded data is too small after basic cleaning ---
            if df is not None:
                initial_customers = df['CustomerID'].nunique()
                # Basic cleaning check (remove NaN CustomerID)
                df_check = df.dropna(subset=['CustomerID']) 
                if df_check['CustomerID'].nunique() < 10: # Threshold for minimum customers
                    print(f"Warning: Loaded data from {source} has too few unique customers ({df_check['CustomerID'].nunique()}) after initial check.")
                    print("Switching to larger synthetic dataset.")
                    df = create_sample_dataset()
                    source = "synthetic"
                    
        return df
    
    except Exception as e:
        print(f"Error in load_data function: {str(e)}")
        # Final fallback if anything else fails
        print("Critical error during loading, generating synthetic dataset as last resort.")
        return create_sample_dataset()

def create_sample_dataset(num_customers=250, avg_transactions_per_cust=80):
    """Creates a larger, more varied and realistic synthetic e-commerce dataset."""
    num_transactions_total = num_customers * avg_transactions_per_cust
    print(f"Generating realistic synthetic dataset: ~{num_customers} customers, ~{num_transactions_total} transactions...")
    np.random.seed(42) # for reproducibility
    
    # --- Define Customer Archetypes & Distribution ---
    # Archetype: (Name, % of Customers, Recency Range (days ago), Freq Range (orders/year), Items/Order Range, Price Pref (0=avg, 1=high, -1=low))
    archetypes = [
        ("⭐ VIP",         0.10, (5, 60),    (20, 60), (3, 8),  1),
        ("💚 Loyal Regular", 0.25, (30, 120),  (10, 30), (2, 6),  0),
        ("👋 New/Recent",  0.20, (0, 90),    (1, 6),   (1, 4),  0),
        ("💰 Occasional Big",0.10, (90, 250),  (2, 8),   (5, 12), 1),
        ("⚠️ At Risk",     0.20, (120, 300), (3, 15),  (2, 5),  0),
        ("👻 Lapsed/Lost", 0.15, (250, 450), (1, 5),   (1, 3), -1) # Increased max recency
    ]
    
    # --- Generate Customers based on Archetypes ---
    customer_data = []
    customer_ids = [f"C{1000+i}" for i in range(num_customers)]
    assigned_archetypes = np.random.choice(
        range(len(archetypes)), 
        num_customers, 
        p=[a[1] for a in archetypes]
    )
    
    for i, cust_id in enumerate(customer_ids):
        archetype_idx = assigned_archetypes[i]
        name, _, r_range, f_range, items_range, price_pref = archetypes[archetype_idx]
        customer_data.append({
            "CustomerID": cust_id,
            "Archetype": name,
            "RecencyRange": r_range,
            "FrequencyRange": f_range,
            "ItemsRange": items_range,
            "PricePref": price_pref
        })
    customers_df = pd.DataFrame(customer_data)
    
    # --- Generate Products with Better Descriptions ---
    product_adjectives = ["Artisan", "Premium", "Organic", "Handcrafted", "Vintage", "Modern", "Classic", "Rustic", "Minimalist", "Luxury", "Sustainable", "Recycled", "Geometric", "Abstract", "Floral"]
    product_nouns = ["Mug", "Bowl", "Vase", "Planter", "Candle Holder", "Picture Frame", "Wall Clock", "Table Lamp", "Coaster Set", "Serving Tray", "Storage Box", "Woven Basket", "Knit Throw", "Cushion", "Tea Set", "Coffee Press", "Bookend Pair", "Desk Organizer", "Wall Art", "Mirror"]
    product_materials = ["Ceramic", "Wood", "Glass", "Metal", "Woven Cotton", "Concrete", "Marble", "Bamboo", "Leather", "Linen", "Copper", "Brass", "Acrylic", "Resin"]
    products = {}
    num_unique_products = 200 # Increased product variety further
    
    # Generate varied prices
    price_ranges = [(5, 25), (25, 75), (75, 150), (150, 300)]
    price_counts = [int(num_unique_products*0.4), int(num_unique_products*0.3), int(num_unique_products*0.2), int(num_unique_products*0.1)]
    product_prices = np.concatenate([np.random.uniform(low, high, count) for (low, high), count in zip(price_ranges, price_counts)])
    if len(product_prices) < num_unique_products:
         product_prices = np.concatenate([product_prices, np.random.uniform(5, 300, num_unique_products - len(product_prices))])
    np.random.shuffle(product_prices)

    product_descriptions_generated = set()
    for i in range(num_unique_products):
        adj = np.random.choice(product_adjectives)
        noun = np.random.choice(product_nouns)
        material = np.random.choice(product_materials) if np.random.rand() < 0.7 else ""
        description = f"{adj} {material} {noun}".replace("  ", " ").strip()
        # Avoid duplicates more robustly
        original_desc = description
        count = 1
        while description in product_descriptions_generated:
            description = f"{original_desc} Style {count}"
            count += 1
        product_descriptions_generated.add(description)
            
        stock_code = f"SKU{10000+i}" 
        products[stock_code] = (description, product_prices[i])
        
    product_codes = list(products.keys())

    # --- Generate Transactions based on Customer Archetypes --- 
    all_transactions = []
    invoice_counter = 530000 # Start from a higher number
    end_simulation_date = datetime(2024, 3, 1) # Simulate data up to beginning of this month for more realistic recency

    for _, customer in customers_df.iterrows():
        cust_id = customer["CustomerID"]
        r_min, r_max = customer["RecencyRange"]
        f_min, f_max = customer["FrequencyRange"]
        items_min, items_max = customer["ItemsRange"]
        price_pref = customer["PricePref"]

        # Determine customer's specific frequency and last purchase date
        customer_freq_orders = np.random.randint(f_min, f_max + 1)
        # Simulate recency with a slight skew towards the lower end of the range for active types
        last_purchase_days_ago = max(1, int(np.random.beta(a=2, b=5) * (r_max - r_min) + r_min)) 
        last_purchase_date = end_simulation_date - pd.Timedelta(days=last_purchase_days_ago)

        # Generate invoice dates spreading backwards from last purchase
        if customer_freq_orders <= 1:
            invoice_dates = [last_purchase_date]
        else:
            # Generate intervals between orders (more frequent = smaller intervals)
            # Use exponential distribution for intervals (more frequent short intervals)
            avg_interval = (365 - last_purchase_days_ago) / customer_freq_orders
            intervals = np.random.exponential(scale=max(5, avg_interval), size=customer_freq_orders - 1)
            # Ensure intervals don't go too far back & add noise
            intervals = np.clip(intervals, 1, 90) + np.random.randint(-3, 4, size=len(intervals)) 
            intervals = np.maximum(1, intervals) # Min 1 day interval
            
            invoice_dates = [last_purchase_date]
            current_date = last_purchase_date
            for interval in intervals:
                 current_date -= pd.Timedelta(days=int(interval))
                 # Ensure we don't go beyond 1.5 years back for simplicity
                 if (end_simulation_date - current_date).days < 540: 
                     invoice_dates.append(current_date)
                 else:
                     break # Stop if going too far back
        
        # Create transactions for each invoice date
        for inv_date in invoice_dates:
            invoice_no = f"INV{invoice_counter}"
            invoice_counter += 1
            num_items = np.random.randint(items_min, items_max + 1)
            
            for _ in range(num_items):
                # Product selection based on price preference
                if price_pref == 1:
                     # Higher chance of picking expensive items
                     probs = product_prices**2 / np.sum(product_prices**2)
                elif price_pref == -1:
                     # Higher chance of picking cheaper items
                     probs = (1/product_prices)**2 / np.sum((1/product_prices)**2)
                else:
                     probs = None # Uniform probability
                
                stock_code = np.random.choice(product_codes, p=probs)
                description, unit_price = products[stock_code]
                quantity = np.random.randint(1, 4) 
                
                all_transactions.append({
                    'InvoiceNo': invoice_no,
                    'StockCode': stock_code,
                    'Description': description,
                    'Quantity': quantity,
                    'InvoiceDate': inv_date.strftime('%Y-%m-%d %H:%M:%S'), 
                    'UnitPrice': round(unit_price * np.random.uniform(0.98, 1.02), 2), 
                    'CustomerID': cust_id,
                    'Country': 'Synthlandia V2' # Updated country name
                })
                
    df = pd.DataFrame(all_transactions)
    df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'])
    df['Quantity'] = df['Quantity'].astype(int)
    df['UnitPrice'] = df['UnitPrice'].astype(float)
    df['CustomerID'] = df['CustomerID'].astype(str)
    
    # Ensure we have roughly the target number of transactions (adjust if needed)
    if len(df) < num_transactions_total * 0.8:
         print(f"Warning: Generated fewer transactions ({len(df)}) than target.")
    elif len(df) > num_transactions_total * 1.2:
         df = df.sample(n=num_transactions_total, random_state=42) # Sample down if too many
         
    print(f"Generated realistic synthetic dataset: {len(df)} transactions, {df['CustomerID'].nunique()} unique customers.")
    return df.sort_values(by='InvoiceDate').reset_index(drop=True)

def clean_data(df):
    """
    Clean and preprocess the e-commerce dataset.
    
    Parameters:
    -----------
    df : pd.DataFrame
        The raw e-commerce dataset
    
    Returns:
    --------
    pd.DataFrame
        The cleaned dataset
    """
    if df is None:
        return None
    
    # Make a copy to avoid modifying the original
    df_clean = df.copy()
    
    # Remove rows with missing CustomerID
    df_clean = df_clean.dropna(subset=['CustomerID'])
    
    # Convert CustomerID to string
    df_clean['CustomerID'] = df_clean['CustomerID'].astype(str)
    
    # Remove canceled orders (indicated by Quantity < 0)
    df_clean = df_clean[df_clean['Quantity'] > 0]
    
    # Remove rows with UnitPrice <= 0
    df_clean = df_clean[df_clean['UnitPrice'] > 0]
    
    # Calculate TotalPrice
    df_clean['TotalPrice'] = df_clean['Quantity'] * df_clean['UnitPrice']
    
    # Remove outliers (optional)
    # Here, we're keeping it simple by removing extreme Quantity values
    q_high = df_clean['Quantity'].quantile(0.99)
    df_clean = df_clean[df_clean['Quantity'] <= q_high]
    
    print(f"After cleaning: {len(df_clean)} rows")
    return df_clean

def create_rfm_features(df, reference_date=None):
    """
    Create RFM (Recency, Frequency, Monetary) features from transaction data.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Cleaned e-commerce dataframe
    reference_date : datetime, optional
        Reference date for recency calculation. If None, uses the maximum date in the data + 1 day.
    
    Returns:
    --------
    pd.DataFrame
        DataFrame with RFM features for each customer
    """
    if df is None:
        return None
    
    # Set reference date for recency calculation
    if reference_date is None:
        reference_date = df['InvoiceDate'].max() + pd.Timedelta(days=1)
    
    # Group by customer and calculate RFM
    rfm = df.groupby('CustomerID').agg({
        'InvoiceDate': lambda x: (reference_date - x.max()).days,  # Recency
        'InvoiceNo': 'nunique',  # Frequency (unique invoices)
        'TotalPrice': 'sum'  # Monetary
    }).reset_index()
    
    # Rename columns
    rfm.columns = ['CustomerID', 'Recency', 'Frequency', 'Monetary']
    
    # Ensure all values are positive
    rfm['Recency'] = rfm['Recency'].apply(lambda x: max(0, x))
    
    print(f"RFM features created for {len(rfm)} customers")
    return rfm

def normalize_data(df):
    """
    Normalize RFM features using StandardScaler.
    
    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame with RFM features
    
    Returns:
    --------
    tuple (pd.DataFrame, StandardScaler)
        Normalized DataFrame and the scaler object
    """
    if df is None:
        return None, None
    
    # Make a copy
    df_normalized = df.copy()
    
    # Select features to normalize
    features = ['Recency', 'Frequency', 'Monetary']
    
    # Create scaler
    scaler = StandardScaler()
    
    # Fit and transform
    df_normalized[features] = scaler.fit_transform(df_normalized[features])
    
    return df_normalized, scaler 