import pandas as pd
import numpy as np
from gensim.models import Word2Vec
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict
import warnings

from src.explorative_data_analysis import prepare_data_pipeline

warnings.filterwarnings('ignore')


# Import from previous scripts if needed
# from data_preparation import prepare_data_pipeline


# ============================================================================
# DATA PREPARATION FOR ITEM2VEC
# ============================================================================

def prepare_transaction_sequences(df, invoice_col='InvoiceNo', item_col='StockCode'):
    """
    Prepare transaction sequences (baskets) for Item2Vec training.
    Each transaction is treated as a "sentence" of products.

    Args:
        df: DataFrame with transactions
        invoice_col: Column name for invoice/transaction ID
        item_col: Column name for item/product ID

    Returns:
        List of lists (sequences of products per transaction)
    """
    print("\n" + "=" * 80)
    print("PREPARING TRANSACTION SEQUENCES FOR ITEM2VEC")
    print("=" * 80)

    # Group products by invoice
    sequences = df.groupby(invoice_col)[item_col].apply(lambda x: [str(item) for item in x]).tolist()

    # Statistics
    sequence_lengths = [len(seq) for seq in sequences]

    print(f"Number of sequences (transactions): {len(sequences)}")
    print(f"Average sequence length: {np.mean(sequence_lengths):.2f} items")
    print(f"Median sequence length: {np.median(sequence_lengths):.0f} items")
    print(f"Min sequence length: {min(sequence_lengths)} items")
    print(f"Max sequence length: {max(sequence_lengths)} items")

    # Distribution
    print(f"\nSequence length distribution:")
    print(
        f"  1 item:      {sum(1 for x in sequence_lengths if x == 1)} ({sum(1 for x in sequence_lengths if x == 1) / len(sequence_lengths) * 100:.1f}%)")
    print(
        f"  2-5 items:   {sum(1 for x in sequence_lengths if 2 <= x <= 5)} ({sum(1 for x in sequence_lengths if 2 <= x <= 5) / len(sequence_lengths) * 100:.1f}%)")
    print(
        f"  6-10 items:  {sum(1 for x in sequence_lengths if 6 <= x <= 10)} ({sum(1 for x in sequence_lengths if 6 <= x <= 10) / len(sequence_lengths) * 100:.1f}%)")
    print(
        f"  11+ items:   {sum(1 for x in sequence_lengths if x > 10)} ({sum(1 for x in sequence_lengths if x > 10) / len(sequence_lengths) * 100:.1f}%)")

    return sequences


def get_product_vocabulary(sequences):
    """
    Get vocabulary (unique products) from sequences.

    Args:
        sequences: List of product sequences

    Returns:
        Set of unique products with statistics
    """
    print("\n" + "=" * 80)
    print("ANALYZING PRODUCT VOCABULARY")
    print("=" * 80)

    # Count product frequencies
    product_freq = defaultdict(int)
    for seq in sequences:
        for product in seq:
            product_freq[product] += 1

    vocab_size = len(product_freq)

    print(f"Vocabulary size: {vocab_size} unique products")
    print(f"Total product occurrences: {sum(product_freq.values())}")

    # Frequency distribution
    frequencies = list(product_freq.values())
    print(f"\nProduct frequency statistics:")
    print(f"  Mean frequency: {np.mean(frequencies):.2f}")
    print(f"  Median frequency: {np.median(frequencies):.0f}")
    print(f"  Min frequency: {min(frequencies)}")
    print(f"  Max frequency: {max(frequencies)}")

    # Top products
    top_products = sorted(product_freq.items(), key=lambda x: x[1], reverse=True)[:10]
    print(f"\nTop 10 most frequent products:")
    for i, (product, freq) in enumerate(top_products, 1):
        print(f"  {i:2d}. Product {product}: {freq} occurrences")

    return product_freq


# ============================================================================
# ITEM2VEC MODEL TRAINING
# ============================================================================

def train_item2vec(sequences, vector_size=100, window=5, min_count=5,
                   workers=4, epochs=10, sg=1):
    """
    Train Item2Vec (Word2Vec) model on product sequences.

    Args:
        sequences: List of product sequences
        vector_size: Dimensionality of product embeddings
        window: Maximum distance between current and predicted product
        min_count: Ignores products with frequency lower than this
        workers: Number of worker threads
        epochs: Number of training iterations
        sg: Training algorithm (1=skip-gram, 0=CBOW)

    Returns:
        Trained Word2Vec model
    """
    print("\n" + "=" * 80)
    print("TRAINING ITEM2VEC MODEL")
    print("=" * 80)

    print(f"Hyperparameters:")
    print(f"  - Vector size (embedding dim): {vector_size}")
    print(f"  - Window size: {window}")
    print(f"  - Min count: {min_count}")
    print(f"  - Epochs: {epochs}")
    print(f"  - Algorithm: {'Skip-gram' if sg == 1 else 'CBOW'}")
    print(f"  - Workers: {workers}")

    print("\nTraining model...")

    model = Word2Vec(
        sentences=sequences,
        vector_size=vector_size,
        window=window,
        min_count=min_count,
        workers=workers,
        epochs=epochs,
        sg=sg,
        seed=42
    )

    print(f"\n✓ Training complete!")
    print(f"  - Vocabulary size: {len(model.wv)}")
    print(f"  - Vector dimensions: {model.wv.vector_size}")

    return model


def evaluate_model_coverage(model, all_products):
    """
    Evaluate how many products are covered by the model.

    Args:
        model: Trained Word2Vec model
        all_products: Set of all product IDs

    Returns:
        Coverage statistics
    """
    print("\n" + "=" * 80)
    print("MODEL COVERAGE ANALYSIS")
    print("=" * 80)

    vocab = set(model.wv.index_to_key)
    all_products_set = set(str(p) for p in all_products)

    covered = vocab & all_products_set
    not_covered = all_products_set - vocab

    coverage = len(covered) / len(all_products_set) * 100

    print(f"Total products: {len(all_products_set)}")
    print(f"Products in model vocabulary: {len(vocab)}")
    print(f"Coverage: {coverage:.2f}%")
    print(f"Products not covered: {len(not_covered)} (filtered by min_count)")

    return {
        'total_products': len(all_products_set),
        'covered_products': len(covered),
        'coverage_pct': coverage,
        'not_covered': not_covered
    }


# ============================================================================
# SIMILARITY AND RECOMMENDATIONS
# ============================================================================

def find_similar_products(model, product_id, topn=10):
    """
    Find most similar products to a given product using cosine similarity.

    Args:
        model: Trained Word2Vec model
        product_id: Product ID to find similarities for
        topn: Number of similar products to return

    Returns:
        List of (product_id, similarity_score) tuples
    """
    product_id = str(product_id)

    if product_id not in model.wv:
        print(f"⚠️  Product {product_id} not in vocabulary")
        return []

    similar_products = model.wv.most_similar(product_id, topn=topn)

    return similar_products


def display_similar_products(model, product_id, full_data, topn=10):
    """
    Display similar products with descriptions.

    Args:
        model: Trained Word2Vec model
        product_id: Product ID
        full_data: DataFrame with product descriptions
        topn: Number of similar products to display
    """
    print("\n" + "=" * 80)
    print(f"SIMILAR PRODUCTS FOR: {product_id}")
    print("=" * 80)

    # Create product description lookup
    product_lookup = full_data[['StockCode', 'Description']].drop_duplicates()
    product_dict = dict(zip(product_lookup['StockCode'].astype(str), product_lookup['Description']))

    # Get source product description
    source_desc = product_dict.get(str(product_id), "Unknown Product")
    print(f"\nSource Product: {source_desc}\n")

    # Find similar products
    similar = find_similar_products(model, product_id, topn=topn)

    if len(similar) == 0:
        return

    print(f"Top {len(similar)} most similar products:\n")
    print("-" * 80)

    for i, (sim_product, score) in enumerate(similar, 1):
        desc = product_dict.get(sim_product, "Unknown Product")
        print(f"{i:2d}. {desc[:60]:<60} (Similarity: {score:.4f})")


def batch_similarity_search(model, product_list, topn=5):
    """
    Find similar products for multiple products at once.

    Args:
        model: Trained Word2Vec model
        product_list: List of product IDs
        topn: Number of similar products per input product

    Returns:
        Dictionary {product_id: [(similar_product, score), ...]}
    """
    print("\n" + "=" * 80)
    print(f"BATCH SIMILARITY SEARCH FOR {len(product_list)} PRODUCTS")
    print("=" * 80)

    results = {}

    for product_id in product_list:
        similar = find_similar_products(model, product_id, topn=topn)
        if len(similar) > 0:
            results[str(product_id)] = similar

    print(f"Found similarities for {len(results)} products")

    return results


# ============================================================================
# ITEM2VEC RECOMMENDATIONS
# ============================================================================

def generate_item2vec_recommendations(model, user_items, topn=10, aggregation='mean'):
    """
    Generate recommendations for a user based on their purchase history.

    Args:
        model: Trained Word2Vec model
        user_items: Set of items the user has purchased
        topn: Number of recommendations
        aggregation: Method to aggregate similarities ('mean', 'max', 'sum')

    Returns:
        List of (product_id, score) tuples
    """
    # Filter user items to only those in vocabulary
    user_items_in_vocab = [str(item) for item in user_items if str(item) in model.wv]

    if len(user_items_in_vocab) == 0:
        return []

    # Get candidate products (similar to user's items)
    candidate_scores = defaultdict(list)

    for item in user_items_in_vocab:
        similar = model.wv.most_similar(item, topn=50)
        for product, score in similar:
            if product not in user_items_in_vocab:  # Don't recommend what user already has
                candidate_scores[product].append(score)

    # Aggregate scores
    final_scores = {}
    for product, scores in candidate_scores.items():
        if aggregation == 'mean':
            final_scores[product] = np.mean(scores)
        elif aggregation == 'max':
            final_scores[product] = np.max(scores)
        elif aggregation == 'sum':
            final_scores[product] = np.sum(scores)

    # Sort and return top N
    sorted_recs = sorted(final_scores.items(), key=lambda x: x[1], reverse=True)

    return sorted_recs[:topn]


def generate_recommendations_for_all_users(model, train_df, test_df, n=10):
    """
    Generate Item2Vec recommendations for all test users.

    Args:
        model: Trained Word2Vec model
        train_df: Training DataFrame
        test_df: Test DataFrame
        n: Number of recommendations per user

    Returns:
        Dictionary {user_id: [(item_id, score), ...]}
    """
    print("\n" + "=" * 80)
    print("GENERATING ITEM2VEC RECOMMENDATIONS FOR ALL USERS")
    print("=" * 80)

    # Get test users
    test_users = test_df['CustomerID'].unique()

    # Get training purchases for each user
    user_items_dict = train_df.groupby('CustomerID')['StockCode'].apply(set).to_dict()

    # Generate recommendations
    all_recommendations = {}
    users_with_recs = 0

    for user_id in test_users:
        if user_id in user_items_dict:
            user_items = user_items_dict[user_id]
            recs = generate_item2vec_recommendations(model, user_items, topn=n)

            if len(recs) > 0:
                all_recommendations[str(user_id)] = recs
                users_with_recs += 1

    print(f"Generated recommendations for {users_with_recs} out of {len(test_users)} users")

    if users_with_recs > 0:
        avg_recs = np.mean([len(recs) for recs in all_recommendations.values()])
        print(f"Average recommendations per user: {avg_recs:.2f}")

    return all_recommendations


# ============================================================================
# EVALUATION
# ============================================================================

def evaluate_item2vec_recommendations(recommendations, test_interactions, k=10):
    """
    Evaluate Item2Vec recommendations using Precision@K and Recall@K.

    Args:
        recommendations: Dict {user_id: [(item_id, score), ...]}
        test_interactions: DataFrame with actual test interactions
        k: Number of recommendations to consider

    Returns:
        Dictionary with evaluation metrics
    """
    print("\n" + "=" * 80)
    print(f"EVALUATING ITEM2VEC RECOMMENDATIONS @K={k}")
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
        recommended_items = set([str(item_id) for item_id, _ in recs[:k]])

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
    results = {
        'precision': np.mean(precisions) if precisions else 0,
        'recall': np.mean(recalls) if recalls else 0,
        'f1': np.mean(f1_scores) if f1_scores else 0,
        'k': k,
        'num_users': len(precisions)
    }

    print(f"Evaluated {results['num_users']} users")
    print(f"Precision@{k}: {results['precision']:.4f}")
    print(f"Recall@{k}:    {results['recall']:.4f}")
    print(f"F1@{k}:        {results['f1']:.4f}")

    return results


def evaluate_at_multiple_k(recommendations, test_interactions, k_values=[1, 5, 10, 15, 20]):
    """
    Evaluate Item2Vec recommendations at multiple K values.

    Args:
        recommendations: Dict {user_id: [(item_id, score), ...]}
        test_interactions: DataFrame with test interactions
        k_values: List of K values to evaluate

    Returns:
        DataFrame with metrics for each K
    """
    print("\n" + "=" * 80)
    print("EVALUATING AT MULTIPLE K VALUES")
    print("=" * 80)

    results = []

    for k in k_values:
        metrics = evaluate_item2vec_recommendations(recommendations, test_interactions, k=k)
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


# ============================================================================
# EMBEDDING VISUALIZATION
# ============================================================================

def visualize_embeddings_tsne(model, n_products=500, perplexity=30,
                              save_path='item2vec_tsne.png'):
    """
    Visualize product embeddings using t-SNE dimensionality reduction.

    Args:
        model: Trained Word2Vec model
        n_products: Number of products to visualize (top N by frequency)
        perplexity: t-SNE perplexity parameter
        save_path: Path to save figure
    """
    print("\n" + "=" * 80)
    print(f"VISUALIZING EMBEDDINGS WITH t-SNE (n={n_products})")
    print("=" * 80)

    # Get product vectors
    product_ids = list(model.wv.index_to_key)[:n_products]
    vectors = np.array([model.wv[product] for product in product_ids])

    print(f"Applying t-SNE to {len(vectors)} product vectors...")
    print(f"Original dimension: {vectors.shape[1]}")

    # Apply t-SNE
    tsne = TSNE(n_components=2, perplexity=perplexity, random_state=42,
                n_iter=1000, verbose=0)
    vectors_2d = tsne.fit_transform(vectors)

    print(f"Reduced to 2D")

    # Create visualization
    fig, ax = plt.subplots(figsize=(14, 10))

    scatter = ax.scatter(vectors_2d[:, 0], vectors_2d[:, 1],
                         alpha=0.6, s=30, c=range(len(vectors_2d)),
                         cmap='viridis', edgecolors='black', linewidth=0.5)

    # Optionally label some points (top 20)
    for i in range(min(20, len(product_ids))):
        ax.annotate(product_ids[i],
                    (vectors_2d[i, 0], vectors_2d[i, 1]),
                    fontsize=8, alpha=0.7)

    ax.set_title(f'Item2Vec Product Embeddings (t-SNE visualization of top {n_products} products)',
                 fontsize=14, fontweight='bold')
    ax.set_xlabel('t-SNE Dimension 1', fontsize=12)
    ax.set_ylabel('t-SNE Dimension 2', fontsize=12)
    ax.grid(True, alpha=0.3)

    plt.colorbar(scatter, ax=ax, label='Product Index')

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Saved visualization: {save_path}")
    plt.show()

    return vectors_2d


def visualize_embeddings_pca(model, n_products=500, save_path='item2vec_pca.png'):
    """
    Visualize product embeddings using PCA dimensionality reduction.

    Args:
        model: Trained Word2Vec model
        n_products: Number of products to visualize
        save_path: Path to save figure
    """
    print("\n" + "=" * 80)
    print(f"VISUALIZING EMBEDDINGS WITH PCA (n={n_products})")
    print("=" * 80)

    # Get product vectors
    product_ids = list(model.wv.index_to_key)[:n_products]
    vectors = np.array([model.wv[product] for product in product_ids])

    print(f"Applying PCA to {len(vectors)} product vectors...")
    print(f"Original dimension: {vectors.shape[1]}")

    # Apply PCA
    pca = PCA(n_components=2, random_state=42)
    vectors_2d = pca.fit_transform(vectors)

    print(f"Reduced to 2D")
    print(f"Explained variance ratio: {pca.explained_variance_ratio_}")
    print(f"Total explained variance: {sum(pca.explained_variance_ratio_):.2%}")

    # Create visualization
    fig, ax = plt.subplots(figsize=(14, 10))

    scatter = ax.scatter(vectors_2d[:, 0], vectors_2d[:, 1],
                         alpha=0.6, s=30, c=range(len(vectors_2d)),
                         cmap='plasma', edgecolors='black', linewidth=0.5)

    # Optionally label some points
    for i in range(min(20, len(product_ids))):
        ax.annotate(product_ids[i],
                    (vectors_2d[i, 0], vectors_2d[i, 1]),
                    fontsize=8, alpha=0.7)

    ax.set_title(f'Item2Vec Product Embeddings (PCA visualization of top {n_products} products)',
                 fontsize=14, fontweight='bold')
    ax.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%} variance)', fontsize=12)
    ax.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%} variance)', fontsize=12)
    ax.grid(True, alpha=0.3)

    plt.colorbar(scatter, ax=ax, label='Product Index')

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Saved visualization: {save_path}")
    plt.show()

    return vectors_2d


def visualize_product_clusters(model, full_data, n_products=200, n_labels=30,
                               save_path='item2vec_clusters.png'):
    """
    Visualize product clusters with actual product descriptions.

    Args:
        model: Trained Word2Vec model
        full_data: DataFrame with product descriptions
        n_products: Number of products to visualize
        n_labels: Number of products to label
        save_path: Path to save figure
    """
    print("\n" + "=" * 80)
    print(f"VISUALIZING PRODUCT CLUSTERS")
    print("=" * 80)

    # Create product description lookup
    product_lookup = full_data[['StockCode', 'Description']].drop_duplicates()
    product_dict = dict(zip(product_lookup['StockCode'].astype(str), product_lookup['Description']))

    # Get product vectors
    product_ids = list(model.wv.index_to_key)[:n_products]
    vectors = np.array([model.wv[product] for product in product_ids])

    # Apply t-SNE
    print("Applying t-SNE...")
    tsne = TSNE(n_components=2, perplexity=30, random_state=42, n_iter=1000, verbose=0)
    vectors_2d = tsne.fit_transform(vectors)

    # Create visualization
    fig, ax = plt.subplots(figsize=(16, 12))

    scatter = ax.scatter(vectors_2d[:, 0], vectors_2d[:, 1],
                         alpha=0.5, s=50, c='steelblue',
                         edgecolors='black', linewidth=0.5)

    # Label top N products with descriptions
    for i in range(min(n_labels, len(product_ids))):
        product_id = product_ids[i]
        desc = product_dict.get(product_id, product_id)
        # Truncate description
        desc_short = desc[:30] + "..." if len(desc) > 30 else desc

        ax.annotate(desc_short,
                    (vectors_2d[i, 0], vectors_2d[i, 1]),
                    fontsize=7, alpha=0.8,
                    bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.3))

    ax.set_title(f'Item2Vec Product Clusters (t-SNE with descriptions)',
                 fontsize=14, fontweight='bold')
    ax.set_xlabel('t-SNE Dimension 1', fontsize=12)
    ax.set_ylabel('t-SNE Dimension 2', fontsize=12)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Saved visualization: {save_path}")
    plt.show()


# ============================================================================
# SAVE RESULTS
# ============================================================================

def save_item2vec_model(model, path='item2vec_model.bin'):
    """
    Save trained Item2Vec model to file.

    Args:
        model: Trained Word2Vec model
        path: Path to save model
    """
    print("\n" + "=" * 80)
    print("SAVING MODEL")
    print("=" * 80)

    model.save(path)
    print(f"Model saved to: {path}")


def load_item2vec_model(path='item2vec_model.bin'):
    """
    Load Item2Vec model from file.

    Args:
        path: Path to model file

    Returns:
        Loaded Word2Vec model
    """
    print("\n" + "=" * 80)
    print("LOADING MODEL")
    print("=" * 80)

    model = Word2Vec.load(path)
    print(f"Model loaded from: {path}")
    print(f"Vocabulary size: {len(model.wv)}")

    return model


def save_embeddings_to_csv(model, output_path='item2vec_embeddings.csv'):
    """
    Export product embeddings to CSV file.

    Args:
        model: Trained Word2Vec model
        output_path: Path to save CSV
    """
    print("\n" + "=" * 80)
    print("EXPORTING EMBEDDINGS TO CSV")
    print("=" * 80)

    # Create DataFrame with embeddings
    product_ids = model.wv.index_to_key
    embeddings = [model.wv[product] for product in product_ids]

    # Create column names
    columns = ['ProductID'] + [f'dim_{i}' for i in range(model.wv.vector_size)]

    # Create DataFrame
    data = []
    for product_id, embedding in zip(product_ids, embeddings):
        row = [product_id] + embedding.tolist()
        data.append(row)

    df = pd.DataFrame(data, columns=columns)

    # Save to CSV
    df.to_csv(output_path, index=False)

    print(f"Saved {len(df)} product embeddings to: {output_path}")
    print(f"Each embedding has {model.wv.vector_size} dimensions")


def save_item2vec_results(model, recommendations, metrics_df, output_prefix='item2vec'):
    """
    Save Item2Vec results to files.

    Args:
        model: Trained Word2Vec model
        recommendations: Dictionary of recommendations
        metrics_df: DataFrame with evaluation metrics
        output_prefix: Prefix for output files
    """
    print("\n" + "=" * 80)
    print("SAVING RESULTS")
    print("=" * 80)

    # Save model
    model_path = f'{output_prefix}_model.bin'
    save_item2vec_model(model, model_path)

    # Save embeddings
    embeddings_path = f'{output_prefix}_embeddings.csv'
    save_embeddings_to_csv(model, embeddings_path)

    # Save metrics
    metrics_path = f'{output_prefix}_metrics.csv'
    metrics_df.to_csv(metrics_path, index=False)
    print(f"Saved metrics: {metrics_path}")

    # Save recommendations sample
    recs_data = []
    for user_id, recs in list(recommendations.items())[:100]:
        for rank, (item_id, score) in enumerate(recs, 1):
            recs_data.append({
                'UserID': user_id,
                'Rank': rank,
                'ItemID': item_id,
                'Score': score
            })

    if len(recs_data) > 0:
        recs_df = pd.DataFrame(recs_data)
        recs_path = f'{output_prefix}_recommendations_sample.csv'
        recs_df.to_csv(recs_path, index=False)
        print(f"Saved recommendations sample: {recs_path}")


# ============================================================================
# COMPLETE ITEM2VEC PIPELINE
# ============================================================================

def item2vec_pipeline(train_df, test_df, full_data,
                      vector_size=100, window=5, min_count=5, epochs=10):
    """
    Complete Item2Vec training and evaluation pipeline.

    Args:
        train_df: Training DataFrame
        test_df: Test DataFrame
        full_data: Full dataset with product descriptions
        vector_size: Dimensionality of embeddings
        window: Context window size
        min_count: Minimum product frequency
        epochs: Number of training epochs

    Returns:
        Dictionary with model, recommendations, and metrics
    """
    print("\n" + "=" * 80)
    print("ITEM2VEC COMPLETE PIPELINE")
    print("=" * 80)

    # Step 1: Prepare sequences
    sequences = prepare_transaction_sequences(train_df)

    # Step 2: Analyze vocabulary
    product_freq = get_product_vocabulary(sequences)

    # Step 3: Train model
    model = train_item2vec(sequences, vector_size=vector_size

                           , window=window, min_count=min_count, epochs=epochs)

    # Step 4: Evaluate coverage
    all_products = set(train_df['StockCode'].unique())
    coverage_stats = evaluate_model_coverage(model, all_products)

    # Step 5: Display sample similarities
    print("\n" + "=" * 80)
    print("SAMPLE PRODUCT SIMILARITIES")
    print("=" * 80)

    # Get top 5 most frequent products
    top_products = sorted(product_freq.items(), key=lambda x: x[1], reverse=True)[:5]

    for product_id, freq in top_products[:3]:  # Show top 3
        display_similar_products(model, product_id, full_data, topn=10)

    # Step 6: Generate recommendations for all users
    recommendations = generate_recommendations_for_all_users(model, train_df, test_df, n=20)

    # Step 7: Evaluate recommendations
    if len(recommendations) > 0:
        metrics_df = evaluate_at_multiple_k(
            recommendations,
            test_df,
            k_values=[1, 5, 10, 15, 20]
        )
    else:
        print("\n⚠️  No recommendations generated. Skipping evaluation.")
        metrics_df = pd.DataFrame()

    # Step 8: Visualize embeddings
    visualize_embeddings_tsne(model, n_products=500, perplexity=30)
    visualize_embeddings_pca(model, n_products=500)
    visualize_product_clusters(model, full_data, n_products=200, n_labels=30)

    # Step 9: Save results
    save_item2vec_results(model, recommendations, metrics_df)

    print("\n" + "=" * 80)
    print("ITEM2VEC PIPELINE COMPLETE!")
    print("=" * 80)

    return {
        'model': model,
        'recommendations': recommendations,
        'metrics': metrics_df,
        'coverage': coverage_stats,
        'sequences': sequences
    }


# ============================================================================
# ADVANCED ANALYSIS FUNCTIONS
# ============================================================================

def analyze_embedding_quality(model, full_data, sample_size=50):
    """
    Analyze the quality of learned embeddings by examining similar products.

    Args:
        model: Trained Word2Vec model
        full_data: DataFrame with product descriptions
        sample_size: Number of products to analyze
    """
    print("\n" + "=" * 80)
    print("ANALYZING EMBEDDING QUALITY")
    print("=" * 80)

    # Create product description lookup
    product_lookup = full_data[['StockCode', 'Description']].drop_duplicates()
    product_dict = dict(zip(product_lookup['StockCode'].astype(str), product_lookup['Description']))

    # Sample random products
    product_ids = list(model.wv.index_to_key)
    sample_products = np.random.choice(product_ids, min(sample_size, len(product_ids)), replace=False)

    print(f"Analyzing {len(sample_products)} random products...\n")

    quality_scores = []

    for product_id in sample_products[:10]:  # Show first 10
        similar = model.wv.most_similar(product_id, topn=5)

        print(f"\nProduct: {product_dict.get(product_id, product_id)[:60]}")
        print("Similar products:")

        for i, (sim_id, score) in enumerate(similar, 1):
            desc = product_dict.get(sim_id, sim_id)[:60]
            print(f"  {i}. {desc} (score: {score:.3f})")

        # Average similarity score as quality metric
        avg_score = np.mean([score for _, score in similar])
        quality_scores.append(avg_score)

    print("\n" + "-" * 80)
    print(f"Average embedding quality score: {np.mean(quality_scores):.4f}")
    print(f"(Higher is better, range 0-1)")


def find_product_analogies(model, product_a, product_b, product_c, topn=5):
    """
    Find product analogies: A is to B as C is to ?
    Example: If "cup" is to "saucer" as "plate" is to ?

    Args:
        model: Trained Word2Vec model
        product_a: First product in analogy
        product_b: Second product in analogy
        product_c: Third product in analogy
        topn: Number of results to return

    Returns:
        List of analogous products
    """
    print("\n" + "=" * 80)
    print(f"FINDING ANALOGY: {product_a} is to {product_b} as {product_c} is to ?")
    print("=" * 80)

    product_a = str(product_a)
    product_b = str(product_b)
    product_c = str(product_c)

    # Check if all products are in vocabulary
    if not all(p in model.wv for p in [product_a, product_b, product_c]):
        print("⚠️  One or more products not in vocabulary")
        return []

    try:
        # Use Word2Vec's analogy function
        # positive=[b, c], negative=[a] gives: b - a + c
        analogies = model.wv.most_similar(positive=[product_b, product_c],
                                          negative=[product_a],
                                          topn=topn)

        print(f"\nTop {len(analogies)} analogous products:")
        for i, (product, score) in enumerate(analogies, 1):
            print(f"  {i}. {product} (score: {score:.4f})")

        return analogies

    except Exception as e:
        print(f"⚠️  Error computing analogy: {str(e)}")
        return []


def cluster_products(model, n_clusters=10, n_products=500):
    """
    Cluster products based on their embeddings using K-Means.

    Args:
        model: Trained Word2Vec model
        n_clusters: Number of clusters
        n_products: Number of products to cluster

    Returns:
        Dictionary with cluster assignments
    """
    print("\n" + "=" * 80)
    print(f"CLUSTERING {n_products} PRODUCTS INTO {n_clusters} CLUSTERS")
    print("=" * 80)

    from sklearn.cluster import KMeans

    # Get product vectors
    product_ids = list(model.wv.index_to_key)[:n_products]
    vectors = np.array([model.wv[product] for product in product_ids])

    print("Running K-Means clustering...")

    # Apply K-Means
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    cluster_labels = kmeans.fit_predict(vectors)

    # Create cluster assignments
    clusters = defaultdict(list)
    for product_id, label in zip(product_ids, cluster_labels):
        clusters[label].append(product_id)

    # Print cluster sizes
    print(f"\nCluster sizes:")
    for cluster_id in sorted(clusters.keys()):
        print(f"  Cluster {cluster_id}: {len(clusters[cluster_id])} products")

    return dict(clusters)


def visualize_clusters_with_labels(model, full_data, n_clusters=8, n_products=400,
                                   save_path='item2vec_kmeans_clusters.png'):
    """
    Visualize product clusters with K-Means and color coding.

    Args:
        model: Trained Word2Vec model
        full_data: DataFrame with product descriptions
        n_clusters: Number of clusters
        n_products: Number of products to visualize
        save_path: Path to save figure
    """
    print("\n" + "=" * 80)
    print(f"VISUALIZING K-MEANS CLUSTERS")
    print("=" * 80)

    from sklearn.cluster import KMeans

    # Get product vectors
    product_ids = list(model.wv.index_to_key)[:n_products]
    vectors = np.array([model.wv[product] for product in product_ids])

    # Apply K-Means
    print("Clustering products...")
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    cluster_labels = kmeans.fit_predict(vectors)

    # Apply t-SNE
    print("Applying t-SNE for visualization...")
    tsne = TSNE(n_components=2, perplexity=30, random_state=42, n_iter=1000, verbose=0)
    vectors_2d = tsne.fit_transform(vectors)

    # Create visualization
    fig, ax = plt.subplots(figsize=(16, 12))

    # Create colormap
    colors = plt.cm.tab10(np.linspace(0, 1, n_clusters))

    # Plot each cluster
    for cluster_id in range(n_clusters):
        mask = cluster_labels == cluster_id
        ax.scatter(vectors_2d[mask, 0], vectors_2d[mask, 1],
                   c=[colors[cluster_id]], label=f'Cluster {cluster_id}',
                   alpha=0.6, s=50, edgecolors='black', linewidth=0.5)

    # Create product description lookup
    product_lookup = full_data[['StockCode', 'Description']].drop_duplicates()
    product_dict = dict(zip(product_lookup['StockCode'].astype(str), product_lookup['Description']))

    # Label a few products from each cluster
    for cluster_id in range(n_clusters):
        mask = cluster_labels == cluster_id
        cluster_indices = np.where(mask)[0][:3]  # First 3 from each cluster

        for idx in cluster_indices:
            product_id = product_ids[idx]
            desc = product_dict.get(product_id, product_id)
            desc_short = desc[:25] + "..." if len(desc) > 25 else desc

            ax.annotate(desc_short,
                        (vectors_2d[idx, 0], vectors_2d[idx, 1]),
                        fontsize=7, alpha=0.7,
                        bbox=dict(boxstyle='round,pad=0.2',
                                  facecolor=colors[cluster_id], alpha=0.3))

    ax.set_title(f'Item2Vec Product Clusters (K-Means with k={n_clusters})',
                 fontsize=14, fontweight='bold')
    ax.set_xlabel('t-SNE Dimension 1', fontsize=12)
    ax.set_ylabel('t-SNE Dimension 2', fontsize=12)
    ax.legend(loc='upper right', fontsize=9)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Saved visualization: {save_path}")
    plt.show()


def compare_recommendation_diversity(recommendations, top_k=10):
    """
    Analyze diversity of recommendations across users.

    Args:
        recommendations: Dict {user_id: [(item_id, score), ...]}
        top_k: Number of top recommendations to consider

    Returns:
        Diversity metrics
    """
    print("\n" + "=" * 80)
    print("ANALYZING RECOMMENDATION DIVERSITY")
    print("=" * 80)

    # Collect all recommended items
    all_recommended = []
    for recs in recommendations.values():
        all_recommended.extend([item for item, _ in recs[:top_k]])

    # Calculate metrics
    unique_items = len(set(all_recommended))
    total_recommendations = len(all_recommended)

    # Item frequency
    item_counts = defaultdict(int)
    for item in all_recommended:
        item_counts[item] += 1

    # Calculate diversity score (higher = more diverse)
    diversity_score = unique_items / total_recommendations if total_recommendations > 0 else 0

    # Calculate Gini coefficient (measures inequality, 0 = perfect equality)
    frequencies = sorted(item_counts.values())
    n = len(frequencies)
    if n == 0:
        gini = 0
    else:
        cumsum = np.cumsum(frequencies)
        gini = (2 * np.sum([(i + 1) * freq for i, freq in enumerate(frequencies)])) / (n * sum(frequencies)) - (
                    n + 1) / n

    print(f"Total recommendations: {total_recommendations}")
    print(f"Unique items recommended: {unique_items}")
    print(f"Diversity score: {diversity_score:.4f} (higher is better)")
    print(f"Gini coefficient: {gini:.4f} (lower means more equal distribution)")

    # Top recommended items
    top_items = sorted(item_counts.items(), key=lambda x: x[1], reverse=True)[:10]
    print(f"\nTop 10 most frequently recommended items:")
    for i, (item, count) in enumerate(top_items, 1):
        print(f"  {i:2d}. {item}: recommended {count} times ({count / len(recommendations) * 100:.1f}% of users)")

    return {
        'unique_items': unique_items,
        'total_recommendations': total_recommendations,
        'diversity_score': diversity_score,
        'gini_coefficient': gini,
        'top_items': top_items
    }


def visualize_metrics_comparison(item2vec_metrics, save_path='item2vec_metrics_plot.png'):
    """
    Visualize Item2Vec metrics across different K values.

    Args:
        item2vec_metrics: DataFrame with metrics
        save_path: Path to save figure
    """
    print("\n" + "=" * 80)
    print("VISUALIZING METRICS")
    print("=" * 80)

    fig, ax = plt.subplots(figsize=(10, 6))

    ax.plot(item2vec_metrics['K'], item2vec_metrics['Precision'],
            marker='o', linewidth=2, label='Precision@K', color='blue')
    ax.plot(item2vec_metrics['K'], item2vec_metrics['Recall'],
            marker='s', linewidth=2, label='Recall@K', color='green')
    ax.plot(item2vec_metrics['K'], item2vec_metrics['F1'],
            marker='^', linewidth=2, label='F1@K', color='red')

    ax.set_xlabel('K (Number of Recommendations)', fontsize=12)
    ax.set_ylabel('Score', fontsize=12)
    ax.set_title('Item2Vec Model Performance', fontsize=14, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.set_xticks(item2vec_metrics['K'])

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Saved visualization: {save_path}")
    plt.show()


# ============================================================================
# INTEGRATION FUNCTION
# ============================================================================

def run_full_item2vec_analysis(train_df, test_df, full_data,
                               vector_size=100, window=5, min_count=5,
                               epochs=10, analyze_clusters=True):
    """
    Run complete Item2Vec analysis with all features.

    Args:
        train_df: Training DataFrame
        test_df: Test DataFrame
        full_data: Full dataset with product descriptions
        vector_size: Embedding dimensionality
        window: Context window size
        min_count: Minimum product frequency
        epochs: Training epochs
        analyze_clusters: Whether to perform clustering analysis

    Returns:
        Complete results dictionary
    """
    print("\n" + "=" * 80)
    print("FULL ITEM2VEC ANALYSIS")
    print("=" * 80)

    # Run main pipeline
    results = item2vec_pipeline(
        train_df=train_df,
        test_df=test_df,
        full_data=full_data,
        vector_size=vector_size,
        window=window,
        min_count=min_count,
        epochs=epochs
    )

    model = results['model']
    recommendations = results['recommendations']

    # Additional analyses
    print("\n" + "=" * 80)
    print("ADDITIONAL ANALYSES")
    print("=" * 80)

    # Analyze embedding quality
    analyze_embedding_quality(model, full_data, sample_size=50)

    # Clustering analysis
    if analyze_clusters:
        clusters = cluster_products(model, n_clusters=8, n_products=500)
        visualize_clusters_with_labels(model, full_data, n_clusters=8, n_products=400)
        results['clusters'] = clusters

    # Diversity analysis
    if len(recommendations) > 0:
        diversity_stats = compare_recommendation_diversity(recommendations, top_k=10)
        results['diversity'] = diversity_stats

    # Visualize metrics
    if len(results['metrics']) > 0:
        visualize_metrics_comparison(results['metrics'])

    print("\n" + "=" * 80)
    print("FULL ITEM2VEC ANALYSIS COMPLETE!")
    print("=" * 80)

    return results


# ============================================================================
# EXAMPLE USAGE
# ============================================================================

if __name__ == "__main__":
    print("\n" + "=" * 80)
    print("ITEM2VEC (WORD2VEC FOR PRODUCTS) MODULE")
    print("=" * 80)
    print("\nThis module provides:")
    print("  1. Transaction sequence preparation")
    print("  2. Item2Vec (Word2Vec) model training")
    print("  3. Product similarity search")
    print("  4. Embedding-based recommendations")
    print("  5. Evaluation metrics (Precision@K, Recall@K, F1@K)")
    print("  6. t-SNE and PCA visualizations")
    print("  7. Product clustering (K-Means)")
    print("  8. Diversity analysis")
    print("  9. Product analogies")
    print("\nReady to use!")
    print("=" * 80)


    data = prepare_data_pipeline('../Online Retail.xlsx', split_method='temporal')
    train_df = data['train_df']
    test_df = data['test_df']
    full_data = data['full_data']


    item2vec_results = run_full_item2vec_analysis(
        train_df=train_df,
        test_df=test_df,
        full_data=full_data,
        vector_size=100,
        window=5,
        min_count=5,
        epochs=10,
        analyze_clusters=True
    )


    model = item2vec_results['model']
    recommendations = item2vec_results['recommendations']
    metrics = item2vec_results['metrics']
    clusters = item2vec_results['clusters']