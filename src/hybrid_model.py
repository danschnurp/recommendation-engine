import pandas as pd
import numpy as np
from collections import defaultdict
import matplotlib.pyplot as plt
import seaborn as sns
import warnings

from src.association_rules_model import association_rules_pipeline
from src.explorative_data_analysis import prepare_data_pipeline
from src.svd_model import svd_pipeline

warnings.filterwarnings('ignore')


# ============================================================================
# POPULARITY BASELINE
# ============================================================================

def calculate_popularity(train_df, method='count'):
    """
    Calculate product popularity from training data.

    Args:
        train_df: Training DataFrame with StockCode
        method: 'count' (frequency), 'quantity' (total sold), or 'revenue'

    Returns:
        Series with popularity scores, sorted descending
    """
    print("\n" + "=" * 80)
    print(f"CALCULATING POPULARITY (method: {method})")
    print("=" * 80)

    if method == 'count':
        popularity = train_df['StockCode'].value_counts()
    elif method == 'quantity':
        popularity = train_df.groupby('StockCode')['Quantity'].sum().sort_values(ascending=False)
    elif method == 'revenue':
        if 'TotalPrice' in train_df.columns:
            popularity = train_df.groupby('StockCode')['TotalPrice'].sum().sort_values(ascending=False)
        else:
            train_df['TotalPrice'] = train_df['Quantity'] * train_df['UnitPrice']
            popularity = train_df.groupby('StockCode')['TotalPrice'].sum().sort_values(ascending=False)
    else:
        raise ValueError("method must be 'count', 'quantity', or 'revenue'")

    print(f"Total products: {len(popularity)}")
    print(f"Top product score: {popularity.iloc[0]:.2f}")
    print(f"Median score: {popularity.median():.2f}")

    return popularity


def get_popularity_recommendations(popularity, user_id=None, user_purchased=None, top_n=10):
    """
    Get top-N popular products, excluding already purchased items.

    Args:
        popularity: Series with popularity scores
        user_id: User ID (for tracking, optional)
        user_purchased: Set of products user already purchased
        top_n: Number of recommendations

    Returns:
        List of (product, score) tuples
    """
    if user_purchased is None:
        user_purchased = set()

    recommendations = []
    for product, score in popularity.items():
        if product not in user_purchased:
            recommendations.append((product, score))
        if len(recommendations) >= top_n:
            break

    return recommendations


def generate_popularity_recommendations_for_all(popularity, train_df, test_df, top_n=10):
    """
    Generate popularity-based recommendations for all test users.

    Args:
        popularity: Series with popularity scores
        train_df: Training DataFrame
        test_df: Test DataFrame
        top_n: Number of recommendations per user

    Returns:
        Dictionary {user_id: [(item_id, score), ...]}
    """
    print("\n" + "=" * 80)
    print("GENERATING POPULARITY RECOMMENDATIONS FOR ALL USERS")
    print("=" * 80)

    # Get user purchase history from training
    user_items = train_df.groupby('CustomerID')['StockCode'].apply(set).to_dict()

    # Get test users
    test_users = test_df['CustomerID'].unique()

    recommendations = {}

    for user_id in test_users:
        user_purchased = user_items.get(user_id, set())
        recs = get_popularity_recommendations(popularity, user_id, user_purchased, top_n)
        recommendations[user_id] = recs

    print(f"Generated recommendations for {len(recommendations)} users")

    return recommendations


# ============================================================================
# EVALUATION FOR POPULARITY
# ============================================================================

def evaluate_popularity_recommendations(recommendations, test_interactions, k=10):
    """
    Evaluate popularity-based recommendations.

    Args:
        recommendations: Dict {user_id: [(item_id, score), ...]}
        test_interactions: Test DataFrame
        k: Number of recommendations to evaluate

    Returns:
        Dictionary with metrics
    """
    print("\n" + "=" * 80)
    print(f"EVALUATING POPULARITY RECOMMENDATIONS @{k}")
    print("=" * 80)

    # Create test dictionary
    test_dict = defaultdict(set)
    for _, row in test_interactions.iterrows():
        test_dict[row['CustomerID']].add(row['StockCode'])

    precisions = []
    recalls = []
    f1_scores = []

    for user_id, recs in recommendations.items():
        if user_id not in test_dict:
            continue

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

    print(f"Precision@{k}: {results['precision']:.4f}")
    print(f"Recall@{k}:    {results['recall']:.4f}")
    print(f"F1@{k}:        {results['f1']:.4f}")

    return results


def evaluate_popularity_at_multiple_k(recommendations, test_interactions, k_values=[1, 5, 10, 15, 20]):
    """
    Evaluate popularity at multiple K values.

    Args:
        recommendations: Dict of recommendations
        test_interactions: Test DataFrame
        k_values: List of K values

    Returns:
        DataFrame with metrics
    """
    results = []

    for k in k_values:
        metrics = evaluate_popularity_recommendations(recommendations, test_interactions, k=k)
        results.append({
            'K': k,
            'Precision': metrics['precision'],
            'Recall': metrics['recall'],
            'F1': metrics['f1']
        })

    return pd.DataFrame(results)


# ============================================================================
# HYBRID MODEL - WEIGHTED COMBINATION
# ============================================================================

def normalize_scores(recommendations):
    """
    Normalize recommendation scores to 0-1 range.

    Args:
        recommendations: Dict {user_id: [(item_id, score), ...]}

    Returns:
        Dict with normalized scores
    """
    normalized = {}

    for user_id, recs in recommendations.items():
        if len(recs) == 0:
            normalized[user_id] = []
            continue

        scores = [score for _, score in recs]
        min_score = min(scores)
        max_score = max(scores)

        if max_score - min_score == 0:
            norm_recs = [(item, 1.0) for item, _ in recs]
        else:
            norm_recs = [(item, (score - min_score) / (max_score - min_score))
                         for item, score in recs]

        normalized[user_id] = norm_recs

    return normalized


def create_hybrid_recommendations(svd_recs, rules_recs, pop_recs,
                                  weights=(0.5, 0.3, 0.2), top_n=10):
    """
    Create hybrid recommendations using weighted combination.

    Args:
        svd_recs: SVD recommendations dict
        rules_recs: Association rules recommendations dict
        pop_recs: Popularity recommendations dict
        weights: Tuple of (svd_weight, rules_weight, pop_weight)
        top_n: Number of final recommendations

    Returns:
        Dictionary with hybrid recommendations
    """
    print("\n" + "=" * 80)
    print("CREATING HYBRID RECOMMENDATIONS")
    print("=" * 80)

    w_svd, w_rules, w_pop = weights
    print(f"Weights: SVD={w_svd}, Rules={w_rules}, Popularity={w_pop}")

    # Normalize all recommendation scores
    svd_norm = normalize_scores(svd_recs)
    rules_norm = normalize_scores(rules_recs)
    pop_norm = normalize_scores(pop_recs)

    # Get all users
    all_users = set(svd_recs.keys()) | set(rules_recs.keys()) | set(pop_recs.keys())

    hybrid_recs = {}

    for user_id in all_users:
        item_scores = defaultdict(float)

        # Add SVD scores
        if user_id in svd_norm:
            for item, score in svd_norm[user_id]:
                item_scores[item] += w_svd * score

        # Add Rules scores
        if user_id in rules_norm:
            for item, score in rules_norm[user_id]:
                item_scores[item] += w_rules * score

        # Add Popularity scores
        if user_id in pop_norm:
            for item, score in pop_norm[user_id]:
                item_scores[item] += w_pop * score

        # Sort by combined score
        sorted_items = sorted(item_scores.items(), key=lambda x: x[1], reverse=True)
        hybrid_recs[user_id] = sorted_items[:top_n]

    print(f"Generated hybrid recommendations for {len(hybrid_recs)} users")

    return hybrid_recs


# ============================================================================
# HYBRID MODEL - SWITCHING STRATEGY
# ============================================================================

def create_switching_hybrid(svd_recs, rules_recs, pop_recs, train_df,
                            high_threshold=10, low_threshold=3, top_n=10):
    """
    Create hybrid using switching strategy based on user activity.

    Args:
        svd_recs: SVD recommendations
        rules_recs: Rules recommendations
        pop_recs: Popularity recommendations
        train_df: Training DataFrame to count user purchases
        high_threshold: Min purchases to use SVD
        low_threshold: Min purchases to use Rules
        top_n: Number of recommendations

    Returns:
        Dictionary with hybrid recommendations
    """
    print("\n" + "=" * 80)
    print("CREATING SWITCHING HYBRID RECOMMENDATIONS")
    print("=" * 80)

    print(f"Strategy: >={high_threshold} purchases → SVD")
    print(f"          {low_threshold}-{high_threshold - 1} purchases → Rules")
    print(f"          <{low_threshold} purchases → Popularity")

    # Count user purchases
    user_purchase_counts = train_df.groupby('CustomerID')['InvoiceNo'].nunique()

    all_users = set(svd_recs.keys()) | set(rules_recs.keys()) | set(pop_recs.keys())

    hybrid_recs = {}
    strategy_counts = {'svd': 0, 'rules': 0, 'popularity': 0}

    for user_id in all_users:
        num_purchases = user_purchase_counts.get(user_id, 0)

        if num_purchases >= high_threshold and user_id in svd_recs:
            hybrid_recs[user_id] = svd_recs[user_id][:top_n]
            strategy_counts['svd'] += 1
        elif num_purchases >= low_threshold and user_id in rules_recs:
            hybrid_recs[user_id] = rules_recs[user_id][:top_n]
            strategy_counts['rules'] += 1
        elif user_id in pop_recs:
            hybrid_recs[user_id] = pop_recs[user_id][:top_n]
            strategy_counts['popularity'] += 1

    print(f"\nStrategy distribution:")
    print(f"  SVD: {strategy_counts['svd']} users")
    print(f"  Rules: {strategy_counts['rules']} users")
    print(f"  Popularity: {strategy_counts['popularity']} users")

    return hybrid_recs


# ============================================================================
# MODEL COMPARISON
# ============================================================================

def compare_all_models(svd_recs, rules_recs, pop_recs, hybrid_recs,
                       test_interactions, k_values=[1, 5, 10, 15, 20]):
    """
    Compare all models at multiple K values.

    Args:
        svd_recs: SVD recommendations
        rules_recs: Rules recommendations
        pop_recs: Popularity recommendations
        hybrid_recs: Hybrid recommendations
        test_interactions: Test DataFrame
        k_values: List of K values

    Returns:
        DataFrame with comparison results
    """
    print("\n" + "=" * 80)
    print("COMPARING ALL MODELS")
    print("=" * 80)

    all_results = []

    models = {
        'SVD': svd_recs,
        'Association Rules': rules_recs,
        'Popularity': pop_recs,
        'Hybrid': hybrid_recs
    }

    test_dict = defaultdict(set)
    for _, row in test_interactions.iterrows():
        test_dict[row['CustomerID']].add(row['StockCode'])

    for model_name, recommendations in models.items():
        print(f"\nEvaluating {model_name}...")

        for k in k_values:
            precisions = []
            recalls = []
            f1_scores = []

            for user_id, recs in recommendations.items():
                if user_id not in test_dict:
                    continue

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

            all_results.append({
                'Model': model_name,
                'K': k,
                'Precision': np.mean(precisions) if precisions else 0,
                'Recall': np.mean(recalls) if recalls else 0,
                'F1': np.mean(f1_scores) if f1_scores else 0
            })

    results_df = pd.DataFrame(all_results)

    print("\n" + "-" * 80)
    print("COMPARISON RESULTS:")
    print(results_df.to_string(index=False))

    return results_df


# ============================================================================
# VISUALIZATION
# ============================================================================

def visualize_model_comparison(comparison_df, save_path='model_comparison.png'):
    """
    Visualize comparison of all models.

    Args:
        comparison_df: DataFrame with comparison results
        save_path: Path to save figure
    """
    print("\n" + "=" * 80)
    print("CREATING COMPARISON VISUALIZATIONS")
    print("=" * 80)

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    models = comparison_df['Model'].unique()
    colors = plt.cm.Set2(range(len(models)))

    # Precision@K
    for i, model in enumerate(models):
        model_data = comparison_df[comparison_df['Model'] == model]
        axes[0].plot(model_data['K'], model_data['Precision'],
                     marker='o', label=model, color=colors[i], linewidth=2)
    axes[0].set_xlabel('K', fontsize=11)
    axes[0].set_ylabel('Precision@K', fontsize=11)
    axes[0].set_title('Precision@K Comparison', fontsize=12, fontweight='bold')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Recall@K
    for i, model in enumerate(models):
        model_data = comparison_df[comparison_df['Model'] == model]
        axes[1].plot(model_data['K'], model_data['Recall'],
                     marker='s', label=model, color=colors[i], linewidth=2)
    axes[1].set_xlabel('K', fontsize=11)
    axes[1].set_ylabel('Recall@K', fontsize=11)
    axes[1].set_title('Recall@K Comparison', fontsize=12, fontweight='bold')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    # F1@K
    for i, model in enumerate(models):
        model_data = comparison_df[comparison_df['Model'] == model]
        axes[2].plot(model_data['K'], model_data['F1'],
                     marker='^', label=model, color=colors[i], linewidth=2)
    axes[2].set_xlabel('K', fontsize=11)
    axes[2].set_ylabel('F1@K', fontsize=11)
    axes[2].set_title('F1@K Comparison', fontsize=12, fontweight='bold')
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Saved visualization: {save_path}")
    plt.show()


def visualize_k10_comparison(comparison_df, save_path='models_at_k10.png'):
    """
    Create bar chart comparing all models at K=10.

    Args:
        comparison_df: DataFrame with comparison results
        save_path: Path to save figure
    """
    k10_data = comparison_df[comparison_df['K'] == 10].copy()

    fig, ax = plt.subplots(figsize=(10, 6))

    x = np.arange(len(k10_data))
    width = 0.25

    ax.bar(x - width, k10_data['Precision'], width, label='Precision@10', alpha=0.8)
    ax.bar(x, k10_data['Recall'], width, label='Recall@10', alpha=0.8)
    ax.bar(x + width, k10_data['F1'], width, label='F1@10', alpha=0.8)

    ax.set_xlabel('Model', fontsize=12)
    ax.set_ylabel('Score', fontsize=12)
    ax.set_title('Model Comparison at K=10', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(k10_data['Model'])
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Saved visualization: {save_path}")
    plt.show()


# ============================================================================
# COMPLETE PIPELINE
# ============================================================================

def baseline_and_hybrid_pipeline(train_df, test_df, svd_recs, rules_recs,
                                 hybrid_type='weighted', weights=(0.5, 0.3, 0.2)):
    """
    Complete pipeline for baseline and hybrid models.

    Args:
        train_df: Training DataFrame
        test_df: Test DataFrame
        svd_recs: SVD recommendations
        rules_recs: Association rules recommendations
        hybrid_type: 'weighted' or 'switching'
        weights: Weights for weighted hybrid (svd, rules, pop)

    Returns:
        Dictionary with all results
    """
    print("\n" + "=" * 80)
    print("BASELINE AND HYBRID PIPELINE")
    print("=" * 80)

    # Calculate popularity
    popularity = calculate_popularity(train_df, method='count')

    # Generate popularity recommendations
    pop_recs = generate_popularity_recommendations_for_all(
        popularity, train_df, test_df, top_n=20
    )

    # Evaluate popularity
    pop_metrics = evaluate_popularity_at_multiple_k(
        pop_recs, test_df, k_values=[1, 5, 10, 15, 20]
    )

    # Create hybrid recommendations
    if hybrid_type == 'weighted':
        hybrid_recs = create_hybrid_recommendations(
            svd_recs, rules_recs, pop_recs,
            weights=weights, top_n=20
        )
    else:
        hybrid_recs = create_switching_hybrid(
            svd_recs, rules_recs, pop_recs, train_df,
            high_threshold=10, low_threshold=3, top_n=20
        )

    # Compare all models
    comparison_df = compare_all_models(
        svd_recs, rules_recs, pop_recs, hybrid_recs,
        test_df, k_values=[1, 5, 10, 15, 20]
    )

    # Visualize
    visualize_model_comparison(comparison_df)
    visualize_k10_comparison(comparison_df)

    # Save results
    comparison_df.to_csv('model_comparison.csv', index=False)
    pop_metrics.to_csv('popularity_metrics.csv', index=False)
    print("\nSaved: model_comparison.csv, popularity_metrics.csv")

    print("\n" + "=" * 80)
    print("BASELINE AND HYBRID PIPELINE COMPLETE!")
    print("=" * 80)

    return {
        'popularity_recs': pop_recs,
        'hybrid_recs': hybrid_recs,
        'comparison': comparison_df,
        'pop_metrics': pop_metrics
    }


# ============================================================================
# EXAMPLE USAGE
# ============================================================================

if __name__ == "__main__":

    data = prepare_data_pipeline('../Online Retail.xlsx',
                                 split_method='user_based',
                                 filter_products=True,
                                 top_n_products=1000
                                 )
    train_df = data['train_df']
    test_df = data['test_df']

    full_data = data['full_data']

    # Run SVD pipeline
    svd_results = svd_pipeline(
        train_interactions=train_df,
        test_interactions=test_df,
        full_data=full_data,
        n_factors=50,
        n_epochs=20,
        rating_column='Quantity'
    )
    rules_results = association_rules_pipeline(
        train_df=train_df,
        test_df=test_df,
        top_n_products=500,
        min_support=0.01,
        min_confidence=0.2,
        min_lift=1.0
    )

    results = baseline_and_hybrid_pipeline(
        train_df=train_df,
        test_df=test_df,
        svd_recs=svd_results['recommendations'],
        rules_recs=rules_results['recommendations'],
        hybrid_type='weighted',
        weights=(0.5, 0.3, 0.2)
    )
