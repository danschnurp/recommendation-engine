import pandas as pd
import numpy as np
from scipy import stats
from collections import defaultdict
import matplotlib.pyplot as plt
import seaborn as sns
import warnings

from src.association_rules_model import association_rules_pipeline
from src.explorative_data_analysis import prepare_data_pipeline
from src.hybrid_model import calculate_popularity, generate_popularity_recommendations_for_all, \
    create_hybrid_recommendations, baseline_and_hybrid_pipeline
from src.svd_model import svd_pipeline

warnings.filterwarnings('ignore')


# ============================================================================
# USER SEGMENTATION FOR A/B TEST
# ============================================================================

def create_ab_groups(test_df, group_ratio=0.5, random_state=42):
    """
    Split test users into A/B groups randomly.

    Args:
        test_df: Test DataFrame
        group_ratio: Proportion for group A (default 0.5 for 50/50)
        random_state: Random seed for reproducibility

    Returns:
        Dictionary with group assignments and metadata
    """
    print("\n" + "=" * 80)
    print("CREATING A/B TEST GROUPS")
    print("=" * 80)

    np.random.seed(random_state)

    # Get unique users
    users = test_df['CustomerID'].unique()
    np.random.shuffle(users)

    # Split users
    split_point = int(len(users) * group_ratio)
    group_a_users = users[:split_point]
    group_b_users = users[split_point:]

    # Create group assignments
    group_assignments = {}
    for user in group_a_users:
        group_assignments[user] = 'A'
    for user in group_b_users:
        group_assignments[user] = 'B'

    print(f"Total users: {len(users)}")
    print(f"Group A (Control): {len(group_a_users)} users ({len(group_a_users) / len(users) * 100:.1f}%)")
    print(f"Group B (Treatment): {len(group_b_users)} users ({len(group_b_users) / len(users) * 100:.1f}%)")

    return {
        'assignments': group_assignments,
        'group_a': set(group_a_users),
        'group_b': set(group_b_users)
    }


def verify_group_balance(ab_groups, test_df, train_df):
    """
    Verify that A/B groups are balanced on key metrics.

    Args:
        ab_groups: Dictionary with group assignments
        test_df: Test DataFrame
        train_df: Training DataFrame

    Returns:
        DataFrame with group statistics
    """
    print("\n" + "=" * 80)
    print("VERIFYING GROUP BALANCE")
    print("=" * 80)

    group_a = ab_groups['group_a']
    group_b = ab_groups['group_b']

    # Calculate statistics for each group
    stats_list = []

    for group_name, group_users in [('A', group_a), ('B', group_b)]:
        # Historical purchases (from train)
        train_group = train_df[train_df['CustomerID'].isin(group_users)]
        purchases_per_user = train_group.groupby('CustomerID')['InvoiceNo'].nunique()

        # Revenue
        if 'TotalPrice' not in train_group.columns:
            train_group['TotalPrice'] = train_group['Quantity'] * train_group['UnitPrice']
        revenue_per_user = train_group.groupby('CustomerID')['TotalPrice'].sum()

        stats_list.append({
            'Group': group_name,
            'Users': len(group_users),
            'Avg_Historical_Purchases': purchases_per_user.mean(),
            'Avg_Historical_Revenue': revenue_per_user.mean(),
            'Median_Historical_Purchases': purchases_per_user.median(),
            'Median_Historical_Revenue': revenue_per_user.median()
        })

    balance_df = pd.DataFrame(stats_list)

    print("\nGroup Statistics:")
    print(balance_df.to_string(index=False))

    # Statistical test for balance
    group_a_purchases = train_df[train_df['CustomerID'].isin(group_a)].groupby('CustomerID')['InvoiceNo'].nunique()
    group_b_purchases = train_df[train_df['CustomerID'].isin(group_b)].groupby('CustomerID')['InvoiceNo'].nunique()

    t_stat, p_value = stats.ttest_ind(group_a_purchases, group_b_purchases)

    print(f"\nBalance Test (Historical Purchases):")
    print(f"  t-statistic: {t_stat:.4f}")
    print(f"  p-value: {p_value:.4f}")
    print(f"  Groups are {'BALANCED' if p_value > 0.05 else 'IMBALANCED'} (α=0.05)")

    return balance_df


# ============================================================================
# ASSIGN RECOMMENDATIONS TO GROUPS
# ============================================================================

def assign_recommendations_to_groups(ab_groups, control_recs, treatment_recs):
    """
    Assign recommendations based on group assignment.

    Args:
        ab_groups: Dictionary with group assignments
        control_recs: Recommendations for control group (e.g., popularity)
        treatment_recs: Recommendations for treatment group (e.g., hybrid)

    Returns:
        Dictionary with assigned recommendations per user
    """
    print("\n" + "=" * 80)
    print("ASSIGNING RECOMMENDATIONS TO GROUPS")
    print("=" * 80)

    assignments = ab_groups['assignments']

    user_recommendations = {}

    for user_id, group in assignments.items():
        if group == 'A':
            user_recommendations[user_id] = {
                'group': 'A',
                'recommendations': control_recs.get(user_id, [])
            }
        else:
            user_recommendations[user_id] = {
                'group': 'B',
                'recommendations': treatment_recs.get(user_id, [])
            }

    print(f"Assigned recommendations to {len(user_recommendations)} users")

    return user_recommendations


# ============================================================================
# SIMULATE BUSINESS METRICS
# ============================================================================

def simulate_click_through_rate(user_recs, test_df, k=10):
    """
    Simulate CTR: user 'clicks' if recommended item is in their test purchases.

    Args:
        user_recs: Dictionary with user recommendations and group
        test_df: Test DataFrame
        k: Number of recommendations to consider

    Returns:
        Dictionary with CTR data per user
    """
    print("\n" + "=" * 80)
    print(f"SIMULATING CLICK-THROUGH RATE (CTR) @{k}")
    print("=" * 80)

    # Create test set dictionary
    test_dict = defaultdict(set)
    for _, row in test_df.iterrows():
        test_dict[row['CustomerID']].add(row['StockCode'])

    ctr_data = []

    for user_id, data in user_recs.items():
        if user_id not in test_dict:
            continue

        recs = data['recommendations'][:k]
        actual = test_dict[user_id]

        # Count clicks (recommendations that user actually bought)
        clicks = sum(1 for item, _ in recs if item in actual)
        impressions = len(recs)

        ctr = clicks / impressions if impressions > 0 else 0

        ctr_data.append({
            'user_id': user_id,
            'group': data['group'],
            'clicks': clicks,
            'impressions': impressions,
            'ctr': ctr
        })

    ctr_df = pd.DataFrame(ctr_data)

    print(f"Simulated CTR for {len(ctr_df)} users")

    return ctr_df


def simulate_conversion_rate(user_recs, test_df, k=10):
    """
    Simulate conversion rate: proportion of recommendations that led to purchase.

    Args:
        user_recs: Dictionary with user recommendations and group
        test_df: Test DataFrame
        k: Number of recommendations

    Returns:
        Dictionary with conversion data per user
    """
    print("\n" + "=" * 80)
    print(f"SIMULATING CONVERSION RATE @{k}")
    print("=" * 80)

    test_dict = defaultdict(set)
    for _, row in test_df.iterrows():
        test_dict[row['CustomerID']].add(row['StockCode'])

    conversion_data = []

    for user_id, data in user_recs.items():
        if user_id not in test_dict:
            continue

        recs = data['recommendations'][:k]
        actual = test_dict[user_id]

        conversions = sum(1 for item, _ in recs if item in actual)
        conversion_rate = conversions / k if k > 0 else 0

        conversion_data.append({
            'user_id': user_id,
            'group': data['group'],
            'conversions': conversions,
            'conversion_rate': conversion_rate
        })

    conversion_df = pd.DataFrame(conversion_data)

    print(f"Simulated conversions for {len(conversion_df)} users")

    return conversion_df


def simulate_revenue(user_recs, test_df, k=10):
    """
    Simulate revenue: sum of prices for recommended items that were purchased.

    Args:
        user_recs: Dictionary with user recommendations and group
        test_df: Test DataFrame
        k: Number of recommendations

    Returns:
        DataFrame with revenue data per user
    """
    print("\n" + "=" * 80)
    print(f"SIMULATING REVENUE @{k}")
    print("=" * 80)

    # Create price lookup
    price_lookup = test_df.groupby('StockCode')['UnitPrice'].mean().to_dict()

    # Create test purchases with quantities
    test_purchases = defaultdict(lambda: defaultdict(float))
    for _, row in test_df.iterrows():
        test_purchases[row['CustomerID']][row['StockCode']] += row['Quantity']

    revenue_data = []

    for user_id, data in user_recs.items():
        if user_id not in test_purchases:
            continue

        recs = data['recommendations'][:k]
        user_test = test_purchases[user_id]

        # Calculate revenue from recommended items that were purchased
        revenue = 0
        items_purchased = 0

        for item, _ in recs:
            if item in user_test:
                quantity = user_test[item]
                price = price_lookup.get(item, 0)
                revenue += quantity * price
                items_purchased += 1

        revenue_data.append({
            'user_id': user_id,
            'group': data['group'],
            'revenue': revenue,
            'items_purchased': items_purchased
        })

    revenue_df = pd.DataFrame(revenue_data)

    print(f"Simulated revenue for {len(revenue_df)} users")

    return revenue_df


# ============================================================================
# STATISTICAL ANALYSIS
# ============================================================================

def analyze_ab_test_results(metric_df, metric_name, group_col='group', value_col=None):
    """
    Perform statistical analysis on A/B test results.

    Args:
        metric_df: DataFrame with metrics per user
        metric_name: Name of the metric being tested
        group_col: Column name for group assignment
        value_col: Column name for metric value (if None, uses metric_name.lower())

    Returns:
        Dictionary with test results
    """
    print("\n" + "=" * 80)
    print(f"ANALYZING A/B TEST: {metric_name}")
    print("=" * 80)

    if value_col is None:
        value_col = metric_name.lower().replace(' ', '_')

    # Split by group
    group_a = metric_df[metric_df[group_col] == 'A'][value_col]
    group_b = metric_df[metric_df[group_col] == 'B'][value_col]

    # Calculate statistics
    mean_a = group_a.mean()
    mean_b = group_b.mean()
    std_a = group_a.std()
    std_b = group_b.std()

    # Calculate uplift
    uplift = ((mean_b - mean_a) / mean_a * 100) if mean_a > 0 else 0

    # Perform t-test
    t_stat, p_value = stats.ttest_ind(group_a, group_b)

    # Calculate confidence interval for difference
    diff = mean_b - mean_a
    se_diff = np.sqrt((std_a ** 2 / len(group_a)) + (std_b ** 2 / len(group_b)))
    ci_95 = (diff - 1.96 * se_diff, diff + 1.96 * se_diff)

    # Determine significance
    is_significant = p_value < 0.05

    print(f"\nGroup A (Control):")
    print(f"  N: {len(group_a)}")
    print(f"  Mean: {mean_a:.4f}")
    print(f"  Std: {std_a:.4f}")

    print(f"\nGroup B (Treatment):")
    print(f"  N: {len(group_b)}")
    print(f"  Mean: {mean_b:.4f}")
    print(f"  Std: {std_b:.4f}")

    print(f"\nTest Results:")
    print(f"  Uplift: {uplift:+.2f}%")
    print(f"  t-statistic: {t_stat:.4f}")
    print(f"  p-value: {p_value:.4f}")
    print(f"  95% CI: [{ci_95[0]:.4f}, {ci_95[1]:.4f}]")
    print(f"  Significant: {'YES' if is_significant else 'NO'} (α=0.05)")

    return {
        'metric': metric_name,
        'mean_a': mean_a,
        'mean_b': mean_b,
        'std_a': std_a,
        'std_b': std_b,
        'uplift': uplift,
        't_statistic': t_stat,
        'p_value': p_value,
        'ci_95': ci_95,
        'is_significant': is_significant
    }


def calculate_power_analysis(group_a, group_b, alpha=0.05):
    """
    Calculate statistical power of the test.

    Args:
        group_a: Data for group A
        group_b: Data for group B
        alpha: Significance level

    Returns:
        Power estimate
    """
    from scipy.stats import norm

    n_a = len(group_a)
    n_b = len(group_b)
    mean_a = group_a.mean()
    mean_b = group_b.mean()
    std_a = group_a.std()
    std_b = group_b.std()

    # Effect size (Cohen's d)
    pooled_std = np.sqrt(((n_a - 1) * std_a ** 2 + (n_b - 1) * std_b ** 2) / (n_a + n_b - 2))
    effect_size = (mean_b - mean_a) / pooled_std if pooled_std > 0 else 0

    # Calculate power
    z_alpha = norm.ppf(1 - alpha / 2)
    se = pooled_std * np.sqrt(1 / n_a + 1 / n_b)
    z_beta = (abs(mean_b - mean_a) - z_alpha * se) / se
    power = norm.cdf(z_beta)

    print(f"\nPower Analysis:")
    print(f"  Effect size (Cohen's d): {effect_size:.4f}")
    print(f"  Statistical power: {power:.4f}")

    return power


# ============================================================================
# COMPREHENSIVE AB TEST
# ============================================================================

def run_complete_ab_test(ab_groups, control_recs, treatment_recs, test_df, k=10):
    """
    Run complete A/B test with multiple metrics.

    Args:
        ab_groups: Dictionary with group assignments
        control_recs: Control group recommendations
        treatment_recs: Treatment group recommendations
        test_df: Test DataFrame
        k: Number of recommendations to evaluate

    Returns:
        Dictionary with all results
    """
    print("\n" + "=" * 80)
    print("RUNNING COMPLETE A/B TEST")
    print("=" * 80)

    # Assign recommendations
    user_recs = assign_recommendations_to_groups(ab_groups, control_recs, treatment_recs)

    # Simulate metrics
    ctr_df = simulate_click_through_rate(user_recs, test_df, k=k)
    conversion_df = simulate_conversion_rate(user_recs, test_df, k=k)
    revenue_df = simulate_revenue(user_recs, test_df, k=k)

    # Analyze each metric
    ctr_results = analyze_ab_test_results(ctr_df, 'CTR', value_col='ctr')
    conversion_results = analyze_ab_test_results(conversion_df, 'Conversion Rate', value_col='conversion_rate')
    revenue_results = analyze_ab_test_results(revenue_df, 'Revenue', value_col='revenue')

    # Calculate power
    calculate_power_analysis(
        ctr_df[ctr_df['group'] == 'A']['ctr'],
        ctr_df[ctr_df['group'] == 'B']['ctr']
    )

    return {
        'ctr_df': ctr_df,
        'conversion_df': conversion_df,
        'revenue_df': revenue_df,
        'ctr_results': ctr_results,
        'conversion_results': conversion_results,
        'revenue_results': revenue_results
    }


# ============================================================================
# VISUALIZATION
# ============================================================================

def visualize_ab_test_results(ab_results, save_path='ab_test_results.png'):
    """
    Visualize A/B test results.

    Args:
        ab_results: Dictionary with A/B test results
        save_path: Path to save figure
    """
    print("\n" + "=" * 80)
    print("CREATING A/B TEST VISUALIZATIONS")
    print("=" * 80)

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # CTR comparison
    ctr_summary = ab_results['ctr_df'].groupby('group')['ctr'].agg(['mean', 'std']).reset_index()
    axes[0, 0].bar(ctr_summary['group'], ctr_summary['mean'],
                   yerr=ctr_summary['std'], capsize=5, alpha=0.7, color=['blue', 'orange'])
    axes[0, 0].set_ylabel('CTR')
    axes[0, 0].set_title(f"CTR Comparison\nUplift: {ab_results['ctr_results']['uplift']:+.2f}%")
    axes[0, 0].grid(True, alpha=0.3, axis='y')

    # Conversion Rate comparison
    conv_summary = ab_results['conversion_df'].groupby('group')['conversion_rate'].agg(['mean', 'std']).reset_index()
    axes[0, 1].bar(conv_summary['group'], conv_summary['mean'],
                   yerr=conv_summary['std'], capsize=5, alpha=0.7, color=['blue', 'orange'])
    axes[0, 1].set_ylabel('Conversion Rate')
    axes[0, 1].set_title(f"Conversion Rate Comparison\nUplift: {ab_results['conversion_results']['uplift']:+.2f}%")
    axes[0, 1].grid(True, alpha=0.3, axis='y')

    # Revenue comparison
    rev_summary = ab_results['revenue_df'].groupby('group')['revenue'].agg(['mean', 'std']).reset_index()
    axes[1, 0].bar(rev_summary['group'], rev_summary['mean'],
                   yerr=rev_summary['std'], capsize=5, alpha=0.7, color=['blue', 'orange'])
    axes[1, 0].set_ylabel('Average Revenue')
    axes[1, 0].set_title(f"Revenue Comparison\nUplift: {ab_results['revenue_results']['uplift']:+.2f}%")
    axes[1, 0].grid(True, alpha=0.3, axis='y')

    # Distribution comparison
    axes[1, 1].hist(ab_results['revenue_df'][ab_results['revenue_df']['group'] == 'A']['revenue'],
                    bins=30, alpha=0.5, label='Group A', color='blue')
    axes[1, 1].hist(ab_results['revenue_df'][ab_results['revenue_df']['group'] == 'B']['revenue'],
                    bins=30, alpha=0.5, label='Group B', color='orange')
    axes[1, 1].set_xlabel('Revenue per User')
    axes[1, 1].set_ylabel('Frequency')
    axes[1, 1].set_title('Revenue Distribution by Group')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Saved visualization: {save_path}")
    plt.show()


def create_summary_table(ab_results):
    """
    Create summary table of A/B test results.

    Args:
        ab_results: Dictionary with A/B test results

    Returns:
        DataFrame with summary
    """
    summary_data = []

    for result_name in ['ctr_results', 'conversion_results', 'revenue_results']:
        result = ab_results[result_name]
        summary_data.append({
            'Metric': result['metric'],
            'Control (A)': f"{result['mean_a']:.4f}",
            'Treatment (B)': f"{result['mean_b']:.4f}",
            'Uplift %': f"{result['uplift']:+.2f}%",
            'p-value': f"{result['p_value']:.4f}",
            'Significant': 'Yes' if result['is_significant'] else 'No'
        })

    summary_df = pd.DataFrame(summary_data)

    print("\n" + "=" * 80)
    print("A/B TEST SUMMARY")
    print("=" * 80)
    print(summary_df.to_string(index=False))

    return summary_df


# ============================================================================
# COMPLETE PIPELINE
# ============================================================================

def ab_test_pipeline(test_df, train_df, control_recs, treatment_recs,
                     group_ratio=0.5, k=10, random_state=42):
    """
    Complete A/B test simulation pipeline.

    Args:
        test_df: Test DataFrame
        train_df: Training DataFrame
        control_recs: Control group recommendations (e.g., popularity)
        treatment_recs: Treatment group recommendations (e.g., hybrid)
        group_ratio: Proportion for group A
        k: Number of recommendations to evaluate
        random_state: Random seed

    Returns:
        Dictionary with all A/B test results
    """
    print("\n" + "=" * 80)
    print("A/B TEST SIMULATION PIPELINE")
    print("=" * 80)

    # Create groups
    ab_groups = create_ab_groups(test_df, group_ratio, random_state)

    # Verify balance
    balance_df = verify_group_balance(ab_groups, test_df, train_df)

    # Run test
    ab_results = run_complete_ab_test(ab_groups, control_recs, treatment_recs, test_df, k=k)

    # Create visualizations
    visualize_ab_test_results(ab_results)

    # Create summary
    summary_df = create_summary_table(ab_results)

    # Save results
    summary_df.to_csv('ab_test_summary.csv', index=False)
    balance_df.to_csv('ab_test_balance.csv', index=False)
    ab_results['revenue_df'].to_csv('ab_test_revenue_detail.csv', index=False)
    print("\nSaved: ab_test_summary.csv, ab_test_balance.csv, ab_test_revenue_detail.csv")

    print("\n" + "=" * 80)
    print("A/B TEST PIPELINE COMPLETE!")
    print("=" * 80)

    return {
        'ab_groups': ab_groups,
        'balance': balance_df,
        'results': ab_results,
        'summary': summary_df
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

    popularity = calculate_popularity(train_df, method='count')


    popularity_recs = generate_popularity_recommendations_for_all(
        popularity, train_df, test_df, top_n=20
    )

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

    hybrid_recs = baseline_and_hybrid_pipeline(
        train_df=train_df,
        test_df=test_df,
        svd_recs=svd_results['recommendations'],
        rules_recs=rules_results['recommendations'],
        hybrid_type='weighted',
        weights=(0.5, 0.3, 0.2)
    )

    ab_test_results = ab_test_pipeline(
        test_df=test_df,
        train_df=train_df,
        control_recs=popularity_recs,
        treatment_recs=hybrid_recs,
        group_ratio=0.5,
        k=10,
        random_state=42
    )
