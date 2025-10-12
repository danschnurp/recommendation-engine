import pandas as pd
import numpy as np
from mlxtend.frequent_patterns import apriori, association_rules
from mlxtend.preprocessing import TransactionEncoder
from collections import defaultdict
import matplotlib.pyplot as plt
import seaborn as sns
import warnings

from src.explorative_data_analysis import prepare_data_pipeline

warnings.filterwarnings('ignore')


# ============================================================================
# BASKET MATRIX CREATION
# ============================================================================

def create_basket_matrix(df, min_support_count=10):
    """
    Create basket matrix for association rules mining.

    Args:
        df: DataFrame with InvoiceNo and StockCode
        min_support_count: Minimum number of times a product must appear

    Returns:
        Basket matrix (one-hot encoded DataFrame)
    """
    print("\n" + "=" * 80)
    print("CREATING BASKET MATRIX")
    print("=" * 80)

    # Filter products that appear at least min_support_count times
    product_counts = df['StockCode'].value_counts()
    popular_products = product_counts[product_counts >= min_support_count].index

    df_filtered = df[df['StockCode'].isin(popular_products)].copy()

    print(f"Original products: {df['StockCode'].nunique()}")
    print(f"Filtered products (min {min_support_count} occurrences): {len(popular_products)}")
    print(f"Transactions retained: {df_filtered['InvoiceNo'].nunique()} / {df['InvoiceNo'].nunique()}")

    # Create basket matrix
    basket = df_filtered.groupby(['InvoiceNo', 'StockCode'])['Quantity'].sum().unstack().reset_index().fillna(
        0).set_index('InvoiceNo')

    # Convert to binary (0/1)
    basket = basket.applymap(lambda x: 1 if x > 0 else 0)

    print(f"Basket matrix shape: {basket.shape}")
    print(f"Sparsity: {(basket == 0).sum().sum() / (basket.shape[0] * basket.shape[1]):.2%}")

    return basket


def create_basket_from_top_products(df, top_n=500):
    """
    Create basket matrix using only top N products.

    Args:
        df: DataFrame with InvoiceNo and StockCode
        top_n: Number of top products to include

    Returns:
        Basket matrix
    """
    print("\n" + "=" * 80)
    print(f"CREATING BASKET MATRIX (TOP {top_n} PRODUCTS)")
    print("=" * 80)

    # Get top products
    top_products = df['StockCode'].value_counts().head(top_n).index

    df_filtered = df[df['StockCode'].isin(top_products)].copy()

    print(f"Using top {top_n} products")
    print(f"Transactions: {df_filtered['InvoiceNo'].nunique()}")

    # Create basket
    basket = df_filtered.groupby(['InvoiceNo', 'StockCode'])['Quantity'].sum().unstack().fillna(0)
    basket = basket.applymap(lambda x: 1 if x > 0 else 0)

    print(f"Basket matrix shape: {basket.shape}")

    return basket


# ============================================================================
# APRIORI ALGORITHM
# ============================================================================

def mine_frequent_itemsets(basket, min_support=0.01, use_colnames=True):
    """
    Mine frequent itemsets using Apriori algorithm.

    Args:
        basket: Binary basket matrix
        min_support: Minimum support threshold
        use_colnames: Use product codes as column names

    Returns:
        DataFrame with frequent itemsets
    """
    print("\n" + "=" * 80)
    print("MINING FREQUENT ITEMSETS")
    print("=" * 80)

    print(f"Min support: {min_support}")
    print(f"Running Apriori algorithm...")

    frequent_itemsets = apriori(basket, min_support=min_support, use_colnames=use_colnames)

    print(f"Found {len(frequent_itemsets)} frequent itemsets")
    print(f"Itemset sizes: {frequent_itemsets['itemsets'].apply(len).value_counts().sort_index().to_dict()}")

    return frequent_itemsets


def generate_association_rules(frequent_itemsets, metric='lift', min_threshold=1.0):
    """
    Generate association rules from frequent itemsets.

    Args:
        frequent_itemsets: DataFrame with frequent itemsets
        metric: Metric to use (lift, confidence, support)
        min_threshold: Minimum threshold for the metric

    Returns:
        DataFrame with association rules
    """
    print("\n" + "=" * 80)
    print("GENERATING ASSOCIATION RULES")
    print("=" * 80)

    print(f"Metric: {metric}, Min threshold: {min_threshold}")

    rules = association_rules(frequent_itemsets, metric=metric, min_threshold=min_threshold)

    if len(rules) == 0:
        print("No rules found! Try lowering the threshold.")
        return rules

    # Sort by lift
    rules = rules.sort_values('lift', ascending=False)

    print(f"Generated {len(rules)} rules")
    print(f"Confidence range: {rules['confidence'].min():.3f} - {rules['confidence'].max():.3f}")
    print(f"Lift range: {rules['lift'].min():.3f} - {rules['lift'].max():.3f}")

    return rules


# ============================================================================
# RULE ANALYSIS
# ============================================================================

def analyze_rules(rules, top_n=20):
    """
    Analyze and display top association rules.

    Args:
        rules: DataFrame with association rules
        top_n: Number of top rules to display
    """
    print("\n" + "=" * 80)
    print(f"TOP {top_n} ASSOCIATION RULES BY LIFT")
    print("=" * 80)

    top_rules = rules.head(top_n)

    for idx, row in top_rules.iterrows():
        antecedents = ', '.join([str(item) for item in row['antecedents']])
        consequents = ', '.join([str(item) for item in row['consequents']])

        print(f"\nRule {idx + 1}:")
        print(f"  If: {antecedents}")
        print(f"  Then: {consequents}")
        print(f"  Support: {row['support']:.4f} | Confidence: {row['confidence']:.3f} | Lift: {row['lift']:.3f}")


def filter_rules_by_product(rules, product_code, side='antecedents'):
    """
    Filter rules containing a specific product.

    Args:
        rules: DataFrame with association rules
        product_code: Product code to filter
        side: 'antecedents', 'consequents', or 'both'

    Returns:
        Filtered rules DataFrame
    """
    if side == 'antecedents':
        filtered = rules[rules['antecedents'].apply(lambda x: product_code in x)]
    elif side == 'consequents':
        filtered = rules[rules['consequents'].apply(lambda x: product_code in x)]
    else:
        filtered = rules[
            rules['antecedents'].apply(lambda x: product_code in x) |
            rules['consequents'].apply(lambda x: product_code in x)
            ]

    print(f"Found {len(filtered)} rules containing {product_code}")

    return filtered.sort_values('lift', ascending=False)


# ============================================================================
# RECOMMENDATIONS FROM RULES
# ============================================================================

def get_recommendations_from_rules(rules, user_products, top_n=10):
    """
    Generate recommendations based on association rules and user's products.

    Args:
        rules: DataFrame with association rules
        user_products: Set or list of products user has purchased
        top_n: Number of recommendations

    Returns:
        List of recommended products with scores
    """
    user_products = set(user_products)
    recommendations = defaultdict(float)

    for _, rule in rules.iterrows():
        antecedents = set(rule['antecedents'])
        consequents = set(rule['consequents'])

        # If user has all antecedents, recommend consequents
        if antecedents.issubset(user_products):
            for item in consequents:
                if item not in user_products:
                    # Score based on lift and confidence
                    score = rule['lift'] * rule['confidence']
                    recommendations[item] = max(recommendations[item], score)

    # Sort by score
    sorted_recs = sorted(recommendations.items(), key=lambda x: x[1], reverse=True)

    return sorted_recs[:top_n]


def generate_recommendations_for_all_users(rules, train_df, test_df, top_n=10):
    """
    Generate recommendations for all users in test set using association rules.

    Args:
        rules: Association rules DataFrame
        train_df: Training data with CustomerID and StockCode
        test_df: Test data
        top_n: Number of recommendations per user

    Returns:
        Dictionary {user_id: [(item_id, score), ...]}
    """
    print("\n" + "=" * 80)
    print(f"GENERATING RECOMMENDATIONS FOR TEST USERS")
    print("=" * 80)

    # Get user purchase history from training data
    user_items = train_df.groupby('CustomerID')['StockCode'].apply(set).to_dict()

    # Get test users
    test_users = test_df['CustomerID'].unique()

    recommendations = {}
    no_recs_count = 0

    for user_id in test_users:
        if user_id in user_items:
            user_products = user_items[user_id]
            recs = get_recommendations_from_rules(rules, user_products, top_n=top_n)

            if len(recs) > 0:
                recommendations[user_id] = recs
            else:
                no_recs_count += 1
        else:
            no_recs_count += 1

    print(f"Generated recommendations for {len(recommendations)} users")
    print(f"Users with no recommendations: {no_recs_count}")

    return recommendations


# ============================================================================
# EVALUATION
# ============================================================================

def evaluate_rule_recommendations(recommendations, test_interactions, k=10):
    """
    Evaluate recommendations from association rules.

    Args:
        recommendations: Dict {user_id: [(item_id, score), ...]}
        test_interactions: Test DataFrame with CustomerID and StockCode
        k: Number of recommendations to evaluate

    Returns:
        Dictionary with metrics
    """
    print("\n" + "=" * 80)
    print(f"EVALUATING RULE-BASED RECOMMENDATIONS @{k}")
    print("=" * 80)

    # Create test set dictionary
    test_dict = defaultdict(set)
    for _, row in test_interactions.iterrows():
        test_dict[row['CustomerID']].add(row['StockCode'])

    precisions = []
    recalls = []
    f1_scores = []

    for user_id, recs in recommendations.items():
        if user_id not in test_dict:
            continue

        # Get top-k recommendations
        recommended = set([item for item, _ in recs[:k]])
        actual = test_dict[user_id]

        if len(recommended) > 0:
            hits = len(recommended & actual)
            precision = hits / len(recommended)
            recall = hits / len(actual) if len(actual) > 0 else 0
            f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

            precisions.append(precision)
            recalls.append(recall)
            f1_scores.append(f1)

    results = {
        'precision': np.mean(precisions) if precisions else 0,
        'recall': np.mean(recalls) if recalls else 0,
        'f1': np.mean(f1_scores) if f1_scores else 0,
        'num_users': len(precisions)
    }

    print(f"Evaluated {results['num_users']} users")
    print(f"Precision@{k}: {results['precision']:.4f}")
    print(f"Recall@{k}:    {results['recall']:.4f}")
    print(f"F1@{k}:        {results['f1']:.4f}")

    return results


def evaluate_at_multiple_k(recommendations, test_interactions, k_values=[1, 5, 10, 15, 20]):
    """
    Evaluate at multiple K values.

    Args:
        recommendations: Dict of recommendations
        test_interactions: Test DataFrame
        k_values: List of K values

    Returns:
        DataFrame with metrics
    """
    results = []

    for k in k_values:
        metrics = evaluate_rule_recommendations(recommendations, test_interactions, k=k)
        results.append({
            'K': k,
            'Precision': metrics['precision'],
            'Recall': metrics['recall'],
            'F1': metrics['f1']
        })

    return pd.DataFrame(results)


# ============================================================================
# VISUALIZATION
# ============================================================================

def visualize_rules(rules, top_n=20, save_path='association_rules.png'):
    """
    Visualize association rules.

    Args:
        rules: DataFrame with association rules
        top_n: Number of top rules to visualize
        save_path: Path to save figure
    """
    print("\n" + "=" * 80)
    print("CREATING VISUALIZATIONS")
    print("=" * 80)

    top_rules = rules.head(top_n)

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    # Support vs Confidence scatter
    axes[0, 0].scatter(rules['support'], rules['confidence'],
                       s=rules['lift'] * 10, alpha=0.5, c=rules['lift'], cmap='viridis')
    axes[0, 0].set_xlabel('Support')
    axes[0, 0].set_ylabel('Confidence')
    axes[0, 0].set_title('Support vs Confidence (size = lift)')
    axes[0, 0].grid(True, alpha=0.3)

    # Top rules by lift
    rule_labels = [f"Rule {i + 1}" for i in range(len(top_rules))]
    axes[0, 1].barh(range(len(top_rules)), top_rules['lift'].values, color='steelblue')
    axes[0, 1].set_yticks(range(len(top_rules)))
    axes[0, 1].set_yticklabels(rule_labels)
    axes[0, 1].set_xlabel('Lift')
    axes[0, 1].set_title(f'Top {top_n} Rules by Lift')
    axes[0, 1].invert_yaxis()

    # Distribution of lift
    axes[1, 0].hist(rules['lift'], bins=50, edgecolor='black', alpha=0.7, color='coral')
    axes[1, 0].set_xlabel('Lift')
    axes[1, 0].set_ylabel('Frequency')
    axes[1, 0].set_title('Distribution of Lift Values')
    axes[1, 0].axvline(1.0, color='red', linestyle='--', label='Lift = 1.0')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)

    # Confidence distribution
    axes[1, 1].hist(rules['confidence'], bins=50, edgecolor='black', alpha=0.7, color='green')
    axes[1, 1].set_xlabel('Confidence')
    axes[1, 1].set_ylabel('Frequency')
    axes[1, 1].set_title('Distribution of Confidence Values')
    axes[1, 1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Saved visualization: {save_path}")
    plt.show()


def visualize_metrics(metrics_df, save_path='rules_metrics.png'):
    """
    Visualize precision and recall at different K values.

    Args:
        metrics_df: DataFrame with K, Precision, Recall
        save_path: Path to save figure
    """
    fig, ax = plt.subplots(figsize=(10, 6))

    ax.plot(metrics_df['K'], metrics_df['Precision'], marker='o', linewidth=2, label='Precision@K')
    ax.plot(metrics_df['K'], metrics_df['Recall'], marker='s', linewidth=2, label='Recall@K')
    ax.plot(metrics_df['K'], metrics_df['F1'], marker='^', linewidth=2, label='F1@K')

    ax.set_xlabel('K (Number of Recommendations)')
    ax.set_ylabel('Score')
    ax.set_title('Association Rules Performance')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_xticks(metrics_df['K'])

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Saved visualization: {save_path}")
    plt.show()


# ============================================================================
# COMPLETE PIPELINE
# ============================================================================

def association_rules_pipeline(train_df, test_df, top_n_products=500,
                               min_support=0.01, min_confidence=0.2, min_lift=1.0):
    """
    Complete association rules pipeline.

    Args:
        train_df: Training DataFrame
        test_df: Test DataFrame
        top_n_products: Number of products to include
        min_support: Minimum support for frequent itemsets
        min_confidence: Minimum confidence for rules
        min_lift: Minimum lift for rules

    Returns:
        Dictionary with rules, recommendations, and metrics
    """
    print("\n" + "=" * 80)
    print("ASSOCIATION RULES PIPELINE")
    print("=" * 80)

    # Create basket matrix
    basket = create_basket_from_top_products(train_df, top_n=top_n_products)

    # Mine frequent itemsets
    frequent_itemsets = mine_frequent_itemsets(basket, min_support=min_support)

    # Generate rules
    rules = generate_association_rules(
        frequent_itemsets,
        metric='confidence',
        min_threshold=min_confidence
    )

    # Filter by lift
    rules = rules[rules['lift'] >= min_lift]
    print(f"Rules after lift filter (>={min_lift}): {len(rules)}")

    # Analyze top rules
    analyze_rules(rules, top_n=20)

    # Generate recommendations
    recommendations = generate_recommendations_for_all_users(
        rules, train_df, test_df, top_n=20
    )

    # Evaluate
    metrics_df = evaluate_at_multiple_k(
        recommendations, test_df,
        k_values=[1, 5, 10, 15, 20]
    )

    # Visualize
    visualize_rules(rules, top_n=20)
    visualize_metrics(metrics_df)

    # Save results
    rules.to_csv('association_rules.csv', index=False)
    metrics_df.to_csv('rules_metrics.csv', index=False)
    print("\nSaved: association_rules.csv, rules_metrics.csv")

    print("\n" + "=" * 80)
    print("ASSOCIATION RULES PIPELINE COMPLETE!")
    print("=" * 80)

    return {
        'rules': rules,
        'recommendations': recommendations,
        'metrics': metrics_df,
        'frequent_itemsets': frequent_itemsets
    }


# ============================================================================
# EXAMPLE USAGE
# ============================================================================

if __name__ == "__main__":
    print("Run data_preparation.py first to get train_df and test_df")


    data = prepare_data_pipeline('../Online Retail.xlsx',
                                 split_method='user_based',
                                 filter_products=True,
                                 top_n_products=1000
                                 )
    train_df = data['train_df']
    test_df = data['test_df']

    results = association_rules_pipeline(
        train_df=train_df,
        test_df=test_df,
        top_n_products=500,
        min_support=0.01,
        min_confidence=0.2,
        min_lift=1.0
    )
