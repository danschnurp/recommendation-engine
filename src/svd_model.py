import pandas as pd
import numpy as np
from surprise import Dataset, Reader, SVD
from surprise.model_selection import cross_validate
from collections import defaultdict
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import warnings

from src.explorative_data_analysis import prepare_data_pipeline

warnings.filterwarnings('ignore')


# ============================================================================
# DATA PREPARATION FOR SURPRISE LIBRARY
# ============================================================================

def prepare_surprise_dataset(interactions_df, rating_column='Quantity'):
    """
    Convert interactions DataFrame to Surprise Dataset format.

    Args:
        interactions_df: DataFrame with CustomerID, StockCode, and rating column
        rating_column: Column to use as rating (Quantity, TotalPrice, or NumPurchases)

    Returns:
        Surprise Dataset object
    """
    print("\n" + "=" * 80)
    print("PREPARING DATA FOR SURPRISE LIBRARY")
    print("=" * 80)

    # Create a copy with required columns
    df = interactions_df[['CustomerID', 'StockCode', rating_column]].copy()

    # Convert to appropriate types
    df['CustomerID'] = df['CustomerID'].astype(str)
    df['StockCode'] = df['StockCode'].astype(str)

    print(f"Using '{rating_column}' as rating")
    print(f"Number of interactions: {len(df)}")
    print(f"Number of users: {df['CustomerID'].nunique()}")
    print(f"Number of items: {df['StockCode'].nunique()}")
    print(f"Rating range: {df[rating_column].min():.2f} to {df[rating_column].max():.2f}")

    # Define the rating scale
    rating_scale = (df[rating_column].min(), df[rating_column].max())

    # Create Reader object
    reader = Reader(rating_scale=rating_scale)

    # Load data into Surprise Dataset
    data = Dataset.load_from_df(df, reader)

    print("Dataset successfully prepared for Surprise")

    return data, df


def create_trainset_testset(train_interactions, test_interactions, rating_column='Quantity'):
    """
    Create train and test sets for Surprise from interaction DataFrames.

    Args:
        train_interactions: Training interactions DataFrame
        test_interactions: Test interactions DataFrame
        rating_column: Column to use as rating

    Returns:
        trainset (Surprise trainset), testset (list of tuples)
    """
    print("\n" + "=" * 80)
    print("CREATING TRAIN AND TEST SETS")
    print("=" * 80)

    # Prepare train data
    train_data, train_df = prepare_surprise_dataset(train_interactions, rating_column)

    # Build full trainset
    trainset = train_data.build_full_trainset()

    # Prepare test data as list of (user, item, rating) tuples
    test_df = test_interactions[['CustomerID', 'StockCode', rating_column]].copy()
    test_df['CustomerID'] = test_df['CustomerID'].astype(str)
    test_df['StockCode'] = test_df['StockCode'].astype(str)

    testset = [(row['CustomerID'], row['StockCode'], row[rating_column])
               for _, row in test_df.iterrows()]

    print(f"Trainset size: {trainset.n_ratings} ratings")
    print(f"Testset size: {len(testset)} ratings")

    return trainset, testset, train_df


# ============================================================================
# SVD MODEL TRAINING
# ============================================================================

def train_svd_model(trainset, n_factors=50, n_epochs=20, lr_all=0.005, reg_all=0.02, verbose=True):
    """
    Train SVD (Singular Value Decomposition) model.

    Args:
        trainset: Surprise trainset object
        n_factors: Number of latent factors
        n_epochs: Number of training epochs
        lr_all: Learning rate for all parameters
        reg_all: Regularization term for all parameters
        verbose: Whether to print training progress

    Returns:
        Trained SVD model
    """
    print("\n" + "=" * 80)
    print("TRAINING SVD MODEL")
    print("=" * 80)

    print(f"Hyperparameters:")
    print(f"  - Latent factors: {n_factors}")
    print(f"  - Epochs: {n_epochs}")
    print(f"  - Learning rate: {lr_all}")
    print(f"  - Regularization: {reg_all}")

    # Initialize SVD model
    svd = SVD(
        n_factors=n_factors,
        n_epochs=n_epochs,
        lr_all=lr_all,
        reg_all=reg_all,
        verbose=verbose
    )

    # Train the model
    print("\nTraining...")
    svd.fit(trainset)

    print("Training complete!")

    return svd


def cross_validate_svd(data, n_factors=50, n_epochs=20, cv=5):
    """
    Perform cross-validation on SVD model.

    Args:
        data: Surprise Dataset object
        n_factors: Number of latent factors
        n_epochs: Number of epochs
        cv: Number of cross-validation folds

    Returns:
        Cross-validation results
    """
    print("\n" + "=" * 80)
    print(f"CROSS-VALIDATION ({cv}-FOLD)")
    print("=" * 80)

    svd = SVD(n_factors=n_factors, n_epochs=n_epochs, verbose=False)

    results = cross_validate(svd, data, measures=['RMSE', 'MAE'], cv=cv, verbose=True)

    print("\nCross-validation results:")
    print(f"  RMSE: {results['test_rmse'].mean():.4f} (+/- {results['test_rmse'].std():.4f})")
    print(f"  MAE:  {results['test_mae'].mean():.4f} (+/- {results['test_mae'].std():.4f})")

    return results


# ============================================================================
# GENERATE PREDICTIONS AND RECOMMENDATIONS
# ============================================================================

def generate_predictions(model, testset):
    """
    Generate predictions for test set.

    Args:
        model: Trained Surprise model
        testset: List of (user, item, true_rating) tuples

    Returns:
        List of Prediction objects
    """
    print("\n" + "=" * 80)
    print("GENERATING PREDICTIONS")
    print("=" * 80)

    predictions = model.test(testset)

    print(f"Generated {len(predictions)} predictions")

    return predictions


def get_top_n_recommendations(predictions, n=10):
    """
    Get top-N recommendations for each user from predictions.

    Args:
        predictions: List of Surprise Prediction objects
        n: Number of recommendations per user

    Returns:
        Dictionary {user_id: [(item_id, estimated_rating), ...]}
    """
    print("\n" + "=" * 80)
    print(f"GENERATING TOP-{n} RECOMMENDATIONS")
    print("=" * 80)

    # Group predictions by user
    top_n = defaultdict(list)

    for prediction in predictions:
        user_id = prediction.uid
        item_id = prediction.iid
        estimated_rating = prediction.est

        top_n[user_id].append((item_id, estimated_rating))

    # Sort and get top N for each user
    for user_id, user_predictions in top_n.items():
        user_predictions.sort(key=lambda x: x[1], reverse=True)
        top_n[user_id] = user_predictions[:n]

    print(f"Generated recommendations for {len(top_n)} users")
    print(f"Average recommendations per user: {np.mean([len(recs) for recs in top_n.values()]):.2f}")

    return dict(top_n)


def recommend_for_user(model, user_id, train_df, all_items, n=10):
    """
    Generate recommendations for a specific user.
    Excludes items the user has already interacted with.

    Args:
        model: Trained Surprise model
        user_id: User ID (string)
        train_df: Training DataFrame with user-item interactions
        all_items: List of all item IDs
        n: Number of recommendations

    Returns:
        List of (item_id, estimated_rating) tuples
    """
    # Get items user has already interacted with
    user_items = set(train_df[train_df['CustomerID'] == user_id]['StockCode'].values)

    # Get items to predict
    items_to_predict = [item for item in all_items if item not in user_items]

    # Generate predictions
    predictions = []
    for item_id in items_to_predict:
        pred = model.predict(user_id, item_id)
        predictions.append((item_id, pred.est))

    # Sort by estimated rating and return top N
    predictions.sort(key=lambda x: x[1], reverse=True)

    return predictions[:n]


# ============================================================================
# EVALUATION METRICS
# ============================================================================

def calculate_precision_recall_at_k(recommendations, test_interactions, k=10):
    """
    Calculate Precision@K and Recall@K for recommendations.

    Args:
        recommendations: Dict {user_id: [(item_id, score), ...]}
        test_interactions: DataFrame with actual test interactions
        k: Number of recommendations to consider

    Returns:
        Dictionary with precision, recall, and f1 scores
    """
    print("\n" + "=" * 80)
    print(f"CALCULATING PRECISION@{k} AND RECALL@{k}")
    print("=" * 80)

    # Convert test interactions to dict of sets
    test_dict = defaultdict(set)
    for _, row in test_interactions.iterrows():
        user_id = str(row['CustomerID'])
        item_id = str(row['StockCode'])
        test_dict[user_id].add(item_id)

    precisions = []
    recalls = []
    f1_scores = []

    for user_id, recs in recommendations.items():
        if user_id not in test_dict:
            continue

        # Get top-k recommended items
        recommended_items = set([item_id for item_id, _ in recs[:k]])

        # Get actual items from test set
        actual_items = test_dict[user_id]

        # Calculate metrics
        if len(recommended_items) > 0:
            hits = len(recommended_items & actual_items)

            precision = hits / len(recommended_items)
            recall = hits / len(actual_items) if len(actual_items) > 0 else 0

            f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

            precisions.append(precision)
            recalls.append(recall)
            f1_scores.append(f1)

    # Calculate averages
    avg_precision = np.mean(precisions) if precisions else 0
    avg_recall = np.mean(recalls) if recalls else 0
    avg_f1 = np.mean(f1_scores) if f1_scores else 0

    print(f"Evaluated {len(precisions)} users")
    print(f"Precision@{k}: {avg_precision:.4f}")
    print(f"Recall@{k}:    {avg_recall:.4f}")
    print(f"F1@{k}:        {avg_f1:.4f}")

    results = {
        'precision': avg_precision,
        'recall': avg_recall,
        'f1': avg_f1,
        'k': k,
        'num_users': len(precisions),
        'precision_list': precisions,
        'recall_list': recalls,
        'f1_list': f1_scores
    }

    return results


def calculate_metrics_at_multiple_k(recommendations, test_interactions, k_values=[1, 5, 10, 15, 20]):
    """
    Calculate precision and recall at multiple K values.

    Args:
        recommendations: Dict {user_id: [(item_id, score), ...]}
        test_interactions: DataFrame with actual test interactions
        k_values: List of K values to evaluate

    Returns:
        DataFrame with metrics for each K
    """
    print("\n" + "=" * 80)
    print("CALCULATING METRICS AT MULTIPLE K VALUES")
    print("=" * 80)

    results = []

    for k in k_values:
        metrics = calculate_precision_recall_at_k(recommendations, test_interactions, k=k)
        results.append({
            'K': k,
            'Precision': metrics['precision'],
            'Recall': metrics['recall'],
            'F1': metrics['f1']
        })

    results_df = pd.DataFrame(results)

    print("\n" + "-" * 80)
    print("Summary:")
    print(results_df.to_string(index=False))

    return results_df


def calculate_coverage(recommendations, all_items):
    """
    Calculate catalog coverage: percentage of items that were recommended.

    Args:
        recommendations: Dict {user_id: [(item_id, score), ...]}
        all_items: Set of all available items

    Returns:
        Coverage percentage
    """
    recommended_items = set()
    for recs in recommendations.values():
        recommended_items.update([item_id for item_id, _ in recs])

    coverage = len(recommended_items) / len(all_items) * 100

    print(f"\nCatalog Coverage: {coverage:.2f}%")
    print(f"Recommended items: {len(recommended_items)} out of {len(all_items)}")

    return coverage


# ============================================================================
# VISUALIZATION
# ============================================================================

def visualize_svd_metrics(metrics_df, save_path='svd_metrics.png'):
    """
    Visualize Precision@K and Recall@K across different K values.

    Args:
        metrics_df: DataFrame with K, Precision, Recall columns
        save_path: Path to save the figure
    """
    print("\n" + "=" * 80)
    print("CREATING VISUALIZATIONS")
    print("=" * 80)

    fig, ax = plt.subplots(1, 1, figsize=(10, 6))

    # Plot Precision and Recall
    ax.plot(metrics_df['K'], metrics_df['Precision'], marker='o', linewidth=2,
            label='Precision@K', color='blue')
    ax.plot(metrics_df['K'], metrics_df['Recall'], marker='s', linewidth=2,
            label='Recall@K', color='green')
    ax.plot(metrics_df['K'], metrics_df['F1'], marker='^', linewidth=2,
            label='F1@K', color='red')

    ax.set_xlabel('K (Number of Recommendations)', fontsize=12)
    ax.set_ylabel('Score', fontsize=12)
    ax.set_title('SVD Model Performance: Precision@K, Recall@K, and F1@K', fontsize=14, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.set_xticks(metrics_df['K'])

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Saved visualization: {save_path}")
    plt.show()


def visualize_recommendation_distribution(recommendations, save_path='svd_distribution.png'):
    """
    Visualize distribution of recommendation scores.

    Args:
        recommendations: Dict {user_id: [(item_id, score), ...]}
        save_path: Path to save the figure
    """
    # Collect all scores
    all_scores = []
    for recs in recommendations.values():
        all_scores.extend([score for _, score in recs])

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Histogram of scores
    axes[0].hist(all_scores, bins=50, edgecolor='black', alpha=0.7, color='steelblue')
    axes[0].set_xlabel('Predicted Rating', fontsize=11)
    axes[0].set_ylabel('Frequency', fontsize=11)
    axes[0].set_title('Distribution of Predicted Ratings', fontsize=12, fontweight='bold')
    axes[0].grid(True, alpha=0.3)

    # Number of recommendations per user
    recs_per_user = [len(recs) for recs in recommendations.values()]
    axes[1].hist(recs_per_user, bins=30, edgecolor='black', alpha=0.7, color='coral')
    axes[1].set_xlabel('Number of Recommendations', fontsize=11)
    axes[1].set_ylabel('Number of Users', fontsize=11)
    axes[1].set_title('Recommendations per User', fontsize=12, fontweight='bold')
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Saved visualization: {save_path}")
    plt.show()


def show_sample_recommendations(recommendations, full_data, n_users=5):
    """
    Display sample recommendations for a few users with product descriptions.

    Args:
        recommendations: Dict {user_id: [(item_id, score), ...]}
        full_data: Full DataFrame with product descriptions
        n_users: Number of users to display
    """
    print("\n" + "=" * 80)
    print(f"SAMPLE RECOMMENDATIONS FOR {n_users} USERS")
    print("=" * 80)

    # Create product description lookup
    product_lookup = full_data[['StockCode', 'Description']].drop_duplicates()
    product_dict = dict(zip(product_lookup['StockCode'].astype(str), product_lookup['Description']))

    # Sample random users
    sample_users = list(recommendations.keys())[:n_users]

    for i, user_id in enumerate(sample_users, 1):
        print(f"\nUser {user_id}:")
        print("-" * 80)

        recs = recommendations[user_id][:10]

        for rank, (item_id, score) in enumerate(recs, 1):
            description = product_dict.get(item_id, "Unknown Product")
            print(f"  {rank:2d}. {description[:60]:<60} (Score: {score:.3f})")


# ============================================================================
# SAVE RESULTS
# ============================================================================

def save_svd_results(model, recommendations, metrics_df, output_prefix='svd'):
    """
    Save SVD model results to files.

    Args:
        model: Trained SVD model
        recommendations: Dictionary of recommendations
        metrics_df: DataFrame with evaluation metrics
        output_prefix: Prefix for output files
    """
    print("\n" + "=" * 80)
    print("SAVING RESULTS")
    print("=" * 80)

    # Save metrics
    metrics_path = f'{output_prefix}_metrics.csv'
    metrics_df.to_csv(metrics_path, index=False)
    print(f"Saved metrics: {metrics_path}")

    # Save recommendations sample (first 100 users)
    recs_data = []
    for user_id, recs in list(recommendations.items())[:100]:
        for rank, (item_id, score) in enumerate(recs, 1):
            recs_data.append({
                'UserID': user_id,
                'Rank': rank,
                'ItemID': item_id,
                'Score': score
            })

    recs_df = pd.DataFrame(recs_data)
    recs_path = f'{output_prefix}_recommendations_sample.csv'
    recs_df.to_csv(recs_path, index=False)
    print(f"Saved recommendations sample: {recs_path}")


# ============================================================================
# COMPLETE SVD PIPELINE
# ============================================================================

def svd_pipeline(train_interactions, test_interactions, full_data,
                 n_factors=50, n_epochs=20, rating_column='Quantity'):
    """
    Complete SVD training and evaluation pipeline.

    Args:
        train_interactions: Training interactions DataFrame
        test_interactions: Test interactions DataFrame
        full_data: Full dataset with product descriptions
        n_factors: Number of latent factors for SVD
        n_epochs: Number of training epochs
        rating_column: Column to use as rating

    Returns:
        Dictionary with model, recommendations, and metrics
    """
    print("\n" + "=" * 80)
    print("SVD COMPLETE PIPELINE")
    print("=" * 80)

    # Step 1: Prepare data
    trainset, testset, train_df = create_trainset_testset(
        train_interactions, test_interactions, rating_column
    )

    # Step 2: Train model
    svd_model = train_svd_model(trainset, n_factors=n_factors, n_epochs=n_epochs)

    # Step 3: Generate predictions
    predictions = generate_predictions(svd_model, testset)

    # Step 4: Get top-N recommendations
    recommendations = get_top_n_recommendations(predictions, n=20)

    # Step 5: Evaluate
    metrics_df = calculate_metrics_at_multiple_k(
        recommendations, test_interactions,
        k_values=[1, 5, 10, 15, 20]
    )

    # Step 6: Calculate coverage
    all_items = set(train_df['StockCode'].unique())
    coverage = calculate_coverage(recommendations, all_items)

    # Step 7: Visualize
    visualize_svd_metrics(metrics_df)
    visualize_recommendation_distribution(recommendations)

    # Step 8: Show samples
    show_sample_recommendations(recommendations, full_data, n_users=5)

    # Step 9: Save results
    save_svd_results(svd_model, recommendations, metrics_df)

    print("\n" + "=" * 80)
    print("SVD PIPELINE COMPLETE!")
    print("=" * 80)

    return {
        'model': svd_model,
        'recommendations': recommendations,
        'metrics': metrics_df,
        'coverage': coverage,
        'predictions': predictions
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
    train_interactions = data['train_interactions']
    test_interactions = data['test_interactions']
    full_data = data['full_data']

    # Run SVD pipeline
    svd_results = svd_pipeline(
        train_interactions=train_interactions,
        test_interactions=test_interactions,
        full_data=full_data,
        n_factors=50,
        n_epochs=20,
        rating_column='Quantity'
    )

    # Access results
    svd_model = svd_results['model']
    recommendations = svd_results['recommendations']
    metrics = svd_results['metrics']