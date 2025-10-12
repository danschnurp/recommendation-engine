import warnings

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.model_selection import train_test_split
warnings.filterwarnings('ignore')

# Set visualization style
sns.set_style('whitegrid')
plt.rcParams['figure.figsize'] = (12, 6)


# ============================================================================
# DATA LOADING AND INITIAL EXPLORATION
# ============================================================================

def load_data(filepath='Online Retail.xlsx'):
    """
    Load the Online Retail dataset from UCI repository.
    
    Args:
        filepath: Path to the Excel file
        
    Returns:
        DataFrame with raw data
    """
    print("Loading dataset...")
    df = pd.read_excel(filepath)
    print(f"Dataset loaded successfully: {df.shape[0]} rows, {df.shape[1]} columns")
    return df


def explore_data(df):
    """
    Perform initial exploratory data analysis.
    
    Args:
        df: Raw DataFrame
    """
    print("\n" + "="*80)
    print("INITIAL DATA EXPLORATION")
    print("="*80)
    
    print("\nDataset Info:")
    print(df.info())
    
    print("\n" + "-"*80)
    print("First few rows:")
    print(df.head())
    
    print("\n" + "-"*80)
    print("Statistical Summary:")
    print(df.describe())
    
    print("\n" + "-"*80)
    print("Missing Values:")
    print(df.isnull().sum())
    
    print("\n" + "-"*80)
    print(f"Unique Customers: {df['CustomerID'].nunique()}")
    print(f"Unique Products: {df['StockCode'].nunique()}")
    print(f"Unique Invoices: {df['InvoiceNo'].nunique()}")
    print(f"Date Range: {df['InvoiceDate'].min()} to {df['InvoiceDate'].max()}")


# ============================================================================
# DATA CLEANING
# ============================================================================

def clean_data(df):
    """
    Clean the dataset by removing invalid entries and returns.
    
    Args:
        df: Raw DataFrame
        
    Returns:
        Cleaned DataFrame
    """
    print("\n" + "="*80)
    print("DATA CLEANING")
    print("="*80)
    
    initial_rows = len(df)
    
    # Remove rows with missing CustomerID
    df = df[df['CustomerID'].notna()].copy()
    print(f"Removed {initial_rows - len(df)} rows with missing CustomerID")
    
    # Remove cancelled orders (InvoiceNo starting with 'C')
    df = df[~df['InvoiceNo'].astype(str).str.startswith('C')].copy()
    print(f"Removed cancelled orders: {initial_rows - len(df)} total rows removed")
    
    # Remove negative quantities (returns)
    df = df[df['Quantity'] > 0].copy()
    print(f"Removed negative quantities")
    
    # Remove negative prices
    df = df[df['UnitPrice'] > 0].copy()
    print(f"Removed negative prices")
    
    # Remove duplicates
    duplicates = df.duplicated().sum()
    df = df.drop_duplicates()
    print(f"Removed {duplicates} duplicate rows")
    
    print(f"\nFinal dataset: {len(df)} rows")
    print(f"Unique Customers: {df['CustomerID'].nunique()}")
    print(f"Unique Products: {df['StockCode'].nunique()}")
    print(f"Unique Invoices: {df['InvoiceNo'].nunique()}")
    
    return df


def create_features(df):
    """
    Create additional features for analysis.
    
    Args:
        df: Cleaned DataFrame
        
    Returns:
        DataFrame with additional features
    """
    print("\n" + "="*80)
    print("FEATURE ENGINEERING")
    print("="*80)
    
    # Create total price column
    df['TotalPrice'] = df['Quantity'] * df['UnitPrice']
    
    # Extract date features
    df['Year'] = df['InvoiceDate'].dt.year
    df['Month'] = df['InvoiceDate'].dt.month
    df['Day'] = df['InvoiceDate'].dt.day
    df['DayOfWeek'] = df['InvoiceDate'].dt.dayofweek
    df['Hour'] = df['InvoiceDate'].dt.hour
    
    # Convert CustomerID to integer for consistency
    df['CustomerID'] = df['CustomerID'].astype(int)
    
    print("Features created:")
    print("- TotalPrice (Quantity × UnitPrice)")
    print("- Temporal features (Year, Month, Day, DayOfWeek, Hour)")
    
    return df


# ============================================================================
# EXPLORATORY VISUALIZATIONS
# ============================================================================

def visualize_basic_stats(df):
    """
    Create basic visualizations of the dataset.
    
    Args:
        df: Cleaned DataFrame with features
    """
    print("\n" + "="*80)
    print("CREATING VISUALIZATIONS")
    print("="*80)
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Top 20 products by quantity sold
    top_products = df.groupby('Description')['Quantity'].sum().sort_values(ascending=False).head(20)
    axes[0, 0].barh(range(len(top_products)), top_products.values)
    axes[0, 0].set_yticks(range(len(top_products)))
    axes[0, 0].set_yticklabels(top_products.index, fontsize=8)
    axes[0, 0].set_xlabel('Total Quantity Sold')
    axes[0, 0].set_title('Top 20 Products by Quantity')
    axes[0, 0].invert_yaxis()
    
    # Distribution of purchases per customer
    purchases_per_customer = df.groupby('CustomerID')['InvoiceNo'].nunique()
    axes[0, 1].hist(purchases_per_customer, bins=50, edgecolor='black', alpha=0.7)
    axes[0, 1].set_xlabel('Number of Purchases')
    axes[0, 1].set_ylabel('Number of Customers')
    axes[0, 1].set_title('Distribution of Purchases per Customer')
    axes[0, 1].set_xlim(0, 100)
    
    # Products per invoice distribution
    products_per_invoice = df.groupby('InvoiceNo')['StockCode'].nunique()
    axes[1, 0].hist(products_per_invoice, bins=50, edgecolor='black', alpha=0.7, color='green')
    axes[1, 0].set_xlabel('Number of Products')
    axes[1, 0].set_ylabel('Number of Invoices')
    axes[1, 0].set_title('Distribution of Products per Invoice')
    axes[1, 0].set_xlim(0, 50)
    
    # Monthly revenue trend
    monthly_revenue = df.groupby(['Year', 'Month'])['TotalPrice'].sum().reset_index()
    monthly_revenue['Date'] = pd.to_datetime(monthly_revenue[['Year', 'Month']].assign(Day=1))
    axes[1, 1].plot(monthly_revenue['Date'], monthly_revenue['TotalPrice'], marker='o', linewidth=2)
    axes[1, 1].set_xlabel('Month')
    axes[1, 1].set_ylabel('Revenue')
    axes[1, 1].set_title('Monthly Revenue Trend')
    axes[1, 1].tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    plt.savefig('eda_overview.png', dpi=300, bbox_inches='tight')
    print("Saved visualization: eda_overview.png")
    plt.show()
    
    # Print summary statistics
    print("\n" + "-"*80)
    print("SUMMARY STATISTICS:")
    print(f"Average purchases per customer: {purchases_per_customer.mean():.2f}")
    print(f"Median purchases per customer: {purchases_per_customer.median():.0f}")
    print(f"Average products per invoice: {products_per_invoice.mean():.2f}")
    print(f"Total revenue: £{df['TotalPrice'].sum():,.2f}")


# ============================================================================
# CREATE USER-ITEM INTERACTIONS
# ============================================================================

def create_user_item_matrix(df):
    """
    Create user-item interaction matrix for collaborative filtering.
    
    Args:
        df: Cleaned DataFrame
        
    Returns:
        DataFrame with CustomerID, StockCode, and interaction metrics
    """
    print("\n" + "="*80)
    print("CREATING USER-ITEM INTERACTIONS")
    print("="*80)
    
    # Aggregate interactions: sum of quantities per customer-product pair
    interactions = df.groupby(['CustomerID', 'StockCode']).agg({
        'Quantity': 'sum',
        'TotalPrice': 'sum',
        'InvoiceNo': 'count'
    }).reset_index()
    
    interactions.columns = ['CustomerID', 'StockCode', 'Quantity', 'TotalPrice', 'NumPurchases']
    
    print(f"Created {len(interactions)} user-item interactions")
    print(f"Sparsity: {1 - (len(interactions) / (df['CustomerID'].nunique() * df['StockCode'].nunique())):.4%}")
    
    return interactions


# ============================================================================
# TRAIN-TEST SPLIT
# ============================================================================

def temporal_train_test_split(df, test_months=1):
    """
    Split data temporally: last N months as test set.
    
    Args:
        df: Cleaned DataFrame
        test_months: Number of months to use as test set
        
    Returns:
        train_df, test_df
    """
    print("\n" + "="*80)
    print("TEMPORAL TRAIN-TEST SPLIT")
    print("="*80)
    
    # Find the cutoff date
    max_date = df['InvoiceDate'].max()
    cutoff_date = max_date - pd.DateOffset(months=test_months)
    
    train_df = df[df['InvoiceDate'] < cutoff_date].copy()
    test_df = df[df['InvoiceDate'] >= cutoff_date].copy()
    
    print(f"Cutoff date: {cutoff_date.date()}")
    print(f"Train set: {len(train_df)} transactions ({train_df['InvoiceDate'].min().date()} to {train_df['InvoiceDate'].max().date()})")
    print(f"Test set: {len(test_df)} transactions ({test_df['InvoiceDate'].min().date()} to {test_df['InvoiceDate'].max().date()})")
    print(f"\nTrain customers: {train_df['CustomerID'].nunique()}")
    print(f"Test customers: {test_df['CustomerID'].nunique()}")
    print(f"Overlapping customers: {len(set(train_df['CustomerID']) & set(test_df['CustomerID']))}")
    
    return train_df, test_df


def user_based_train_test_split(df, test_size=0.2, random_state=42):
    """
    Split data by user: for each user, randomly assign transactions to train/test.
    
    Args:
        df: Cleaned DataFrame
        test_size: Proportion of each user's transactions for test
        random_state: Random seed
        
    Returns:
        train_df, test_df
    """
    print("\n" + "="*80)
    print("USER-BASED TRAIN-TEST SPLIT")
    print("="*80)
    
    train_list = []
    test_list = []
    
    for customer_id, customer_data in df.groupby('CustomerID'):
        # Only split if customer has at least 5 transactions
        if len(customer_data) >= 5:
            train, test = train_test_split(
                customer_data, 
                test_size=test_size, 
                random_state=random_state
            )
            train_list.append(train)
            test_list.append(test)
        else:
            # If too few transactions, put all in train
            train_list.append(customer_data)
    
    train_df = pd.concat(train_list, ignore_index=True)
    test_df = pd.concat(test_list, ignore_index=True)
    
    print(f"Train set: {len(train_df)} transactions")
    print(f"Test set: {len(test_df)} transactions")
    print(f"Train customers: {train_df['CustomerID'].nunique()}")
    print(f"Test customers: {test_df['CustomerID'].nunique()}")
    
    return train_df, test_df


# ============================================================================
# FILTER TOP PRODUCTS
# ============================================================================

def filter_top_products(df, top_n=500):
    """
    Filter dataset to include only top N most popular products.
    This speeds up computation significantly.
    
    Args:
        df: DataFrame
        top_n: Number of top products to keep
        
    Returns:
        Filtered DataFrame
    """
    print("\n" + "="*80)
    print(f"FILTERING TOP {top_n} PRODUCTS")
    print("="*80)
    
    # Get top products by total quantity sold
    top_products = df.groupby('StockCode')['Quantity'].sum().sort_values(ascending=False).head(top_n).index
    
    df_filtered = df[df['StockCode'].isin(top_products)].copy()
    
    print(f"Original products: {df['StockCode'].nunique()}")
    print(f"Filtered products: {df_filtered['StockCode'].nunique()}")
    print(f"Original transactions: {len(df)}")
    print(f"Filtered transactions: {len(df_filtered)} ({len(df_filtered)/len(df)*100:.1f}% retained)")
    
    return df_filtered


# ============================================================================
# MAIN EXECUTION PIPELINE
# ============================================================================

def prepare_data_pipeline(filepath='Online Retail.xlsx', 
                         split_method='temporal',
                         filter_products=True,
                          visualize=False,
                         top_n_products=500):
    """
    Complete data preparation pipeline.
    
    Args:
        filepath: Path to dataset
        split_method: 'temporal' or 'user_based'
        filter_products: Whether to filter to top N products
        top_n_products: Number of products to keep if filtering
        
    Returns:
        Dictionary containing all prepared datasets
    """
    print("\n" + "="*80)
    print("STARTING DATA PREPARATION PIPELINE")
    print("="*80)
    
    # Load data
    df_raw = load_data(filepath)
    
    # Explore
    explore_data(df_raw)
    
    # Clean
    df_clean = clean_data(df_raw)
    
    # Create features
    df = create_features(df_clean)
    if visualize:
        visualize_basic_stats(df)
    
    # Filter products if requested
    if filter_products:
        df = filter_top_products(df, top_n=top_n_products)
    
    # Split data
    if split_method == 'temporal':
        train_df, test_df = temporal_train_test_split(df, test_months=1)
    else:
        train_df, test_df = user_based_train_test_split(df, test_size=0.2)
    
    # Create interaction matrices
    train_interactions = create_user_item_matrix(train_df)
    test_interactions = create_user_item_matrix(test_df)
    
    print("\n" + "="*80)
    print("DATA PREPARATION COMPLETE")
    print("="*80)
    
    return {
        'full_data': df,
        'train_df': train_df,
        'test_df': test_df,
        'train_interactions': train_interactions,
        'test_interactions': test_interactions
    }



if __name__ == "__main__":
    # Run the complete pipeline
    data = prepare_data_pipeline(
        filepath='Online Retail.xlsx',
        split_method='user_based',
        filter_products=True,
        visualize=True,
        top_n_products=1000
    )
    
    # Access prepared datasets
    train_df = data['train_df']
    test_df = data['test_df']
    train_interactions = data['train_interactions']
    test_interactions = data['test_interactions']
    
    print("\n" + "="*80)
    print("Ready for model training!")
    print("="*80)
