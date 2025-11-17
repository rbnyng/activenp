"""
Analyze sweep experiment results.

This script aggregates results across multiple runs and generates:
- Learning curves with error bands (mean ± SEM)
- Sample efficiency analysis
- Statistical significance tests
- Publication-ready plots
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

import argparse
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from typing import Dict, List, Tuple
from collections import defaultdict


# Set publication-quality plot style
sns.set_style("whitegrid")
sns.set_context("paper", font_scale=1.5)
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['font.family'] = 'sans-serif'


def load_sweep_results(sweep_dir: Path) -> Dict:
    """
    Load all results from sweep directory.

    Returns:
        Dict with structure:
        {
            seed_size: {
                strategy: {
                    'runs': [history1, history2, ...],  # One per random seed
                    'configs': [config1, config2, ...]
                }
            }
        }
    """
    results = defaultdict(lambda: defaultdict(lambda: {'runs': [], 'configs': []}))

    # Iterate through seed_X/run_Y/strategy directories
    for seed_dir in sorted(sweep_dir.glob('seed_*')):
        seed_size = int(seed_dir.name.split('_')[1])

        for run_dir in sorted(seed_dir.glob('run_*')):
            # Load results.json from this run
            results_file = run_dir / 'results.json'

            if not results_file.exists():
                print(f"Warning: Missing results in {run_dir}")
                continue

            with open(results_file) as f:
                run_data = json.load(f)

            # Extract histories for each strategy
            for strategy, history in run_data['histories'].items():
                results[seed_size][strategy]['runs'].append(history)
                results[seed_size][strategy]['configs'].append(run_data['config'])

    # Convert to regular dict
    return dict(results)


def aggregate_runs(runs: List[Dict]) -> Dict:
    """
    Aggregate multiple runs (different random seeds) into mean and SEM.

    Args:
        runs: List of history dicts

    Returns:
        Dict with arrays for mean and SEM of each metric
    """
    if not runs:
        return {}

    # Get all metrics
    metrics = [k for k in runs[0].keys() if k != 'selected_indices']

    aggregated = {}

    for metric in metrics:
        # Stack all runs (may have different lengths)
        # Find minimum length
        min_length = min(len(run[metric]) for run in runs)

        # Truncate all runs to minimum length and convert to float (handle string values from JSON)
        values = np.array([[float(v) for v in run[metric][:min_length]] for run in runs])

        aggregated[f'{metric}_mean'] = values.mean(axis=0)
        aggregated[f'{metric}_sem'] = values.std(axis=0) / np.sqrt(len(runs))
        aggregated[f'{metric}_std'] = values.std(axis=0)
        aggregated[f'{metric}_all'] = values  # Keep all runs for stats

    return aggregated


def plot_learning_curves(
    results: Dict,
    output_dir: Path,
    metric: str = 'test_rmse',
    metric_label: str = 'Test RMSE (log space)'
):
    """
    Plot learning curves for all seed sizes and strategies.

    Args:
        results: Results dict from load_sweep_results
        output_dir: Output directory
        metric: Metric to plot (e.g., 'test_rmse', 'test_r2')
        metric_label: Label for y-axis
    """
    seed_sizes = sorted(results.keys())
    strategies = ['random', 'uncertainty', 'spatial', 'hybrid']
    colors = {
        'random': 'gray',
        'uncertainty': 'red',
        'spatial': 'blue',
        'hybrid': 'green'
    }

    # Create subplots for each seed size
    n_seeds = len(seed_sizes)
    n_cols = min(3, n_seeds)
    n_rows = (n_seeds + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(6*n_cols, 5*n_rows))
    if n_seeds == 1:
        axes = np.array([axes])
    axes = axes.flatten()

    for idx, seed_size in enumerate(seed_sizes):
        ax = axes[idx]

        for strategy in strategies:
            if strategy not in results[seed_size]:
                continue

            # Aggregate runs
            runs = results[seed_size][strategy]['runs']
            agg = aggregate_runs(runs)

            if not agg:
                continue

            n_train = agg['n_train_mean']
            metric_mean = agg[f'{metric}_mean']
            metric_sem = agg[f'{metric}_sem']

            # Plot mean with error band
            ax.plot(
                n_train, metric_mean,
                marker='o',
                label=f'{strategy.capitalize()} (n={len(runs)})',
                color=colors.get(strategy, 'black'),
                linewidth=2,
                markersize=6
            )

            # Error band (mean ± SEM)
            ax.fill_between(
                n_train,
                metric_mean - metric_sem,
                metric_mean + metric_sem,
                color=colors.get(strategy, 'black'),
                alpha=0.2
            )

        ax.set_xlabel('Number of Training Samples', fontweight='bold')
        ax.set_ylabel(metric_label, fontweight='bold')
        ax.set_title(f'Initial Seed Size: {seed_size}', fontweight='bold')
        ax.legend(loc='best', fontsize=10)
        ax.grid(True, alpha=0.3)

    # Remove empty subplots
    for idx in range(n_seeds, len(axes)):
        fig.delaxes(axes[idx])

    plt.tight_layout()
    output_path = output_dir / f'learning_curves_{metric}.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved: {output_path}")
    plt.close()


def compute_sample_efficiency(
    results: Dict,
    target_rmse: float = 0.5
) -> pd.DataFrame:
    """
    Compute sample efficiency: samples needed to reach target RMSE.

    Args:
        results: Results dict
        target_rmse: Target RMSE threshold

    Returns:
        DataFrame with columns: seed_size, strategy, samples_needed_mean, samples_needed_sem
    """
    efficiency_data = []

    for seed_size in sorted(results.keys()):
        for strategy, data in results[seed_size].items():
            runs = data['runs']

            samples_needed_list = []

            for run in runs:
                # Convert to float to handle string values from JSON
                n_train = np.array([float(x) for x in run['n_train']])
                test_rmse = np.array([float(x) for x in run['test_rmse']])

                # Find first point where RMSE < target
                idx = np.where(test_rmse < target_rmse)[0]

                if len(idx) > 0:
                    samples_needed = n_train[idx[0]]
                else:
                    # Never reached target, use max samples
                    samples_needed = n_train[-1] if len(n_train) > 0 else np.nan

                samples_needed_list.append(samples_needed)

            # Compute stats
            samples_needed_list = [x for x in samples_needed_list if not np.isnan(x)]

            if samples_needed_list:
                efficiency_data.append({
                    'seed_size': seed_size,
                    'strategy': strategy,
                    'samples_needed_mean': np.mean(samples_needed_list),
                    'samples_needed_sem': np.std(samples_needed_list) / np.sqrt(len(samples_needed_list)),
                    'n_runs_reached': len(samples_needed_list),
                    'n_runs_total': len(runs)
                })

    return pd.DataFrame(efficiency_data)


def plot_sample_efficiency(
    efficiency_df: pd.DataFrame,
    output_dir: Path
):
    """Plot sample efficiency comparison."""
    if efficiency_df.empty:
        print("No efficiency data to plot")
        return

    seed_sizes = sorted(efficiency_df['seed_size'].unique())
    strategies = ['random', 'uncertainty', 'spatial', 'hybrid']
    colors = {
        'random': 'gray',
        'uncertainty': 'red',
        'spatial': 'blue',
        'hybrid': 'green'
    }

    fig, ax = plt.subplots(figsize=(10, 6))

    x = np.arange(len(seed_sizes))
    width = 0.2

    for i, strategy in enumerate(strategies):
        strategy_data = efficiency_df[efficiency_df['strategy'] == strategy]

        if strategy_data.empty:
            continue

        # Match seed sizes
        means = []
        sems = []
        for seed_size in seed_sizes:
            row = strategy_data[strategy_data['seed_size'] == seed_size]
            if not row.empty:
                means.append(row.iloc[0]['samples_needed_mean'])
                sems.append(row.iloc[0]['samples_needed_sem'])
            else:
                means.append(np.nan)
                sems.append(0)

        # Plot bars
        ax.bar(
            x + i * width,
            means,
            width,
            yerr=sems,
            label=strategy.capitalize(),
            color=colors.get(strategy, 'black'),
            alpha=0.8,
            capsize=5
        )

    ax.set_xlabel('Initial Seed Size', fontweight='bold')
    ax.set_ylabel('Samples Needed to Reach Target RMSE', fontweight='bold')
    ax.set_title('Sample Efficiency Comparison', fontweight='bold')
    ax.set_xticks(x + width * 1.5)
    ax.set_xticklabels(seed_sizes)
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    output_path = output_dir / 'sample_efficiency.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved: {output_path}")
    plt.close()


def plot_cold_start_advantage(
    results: Dict,
    output_dir: Path,
    metric: str = 'test_rmse'
):
    """
    Plot the advantage of uncertainty sampling over random at different seed sizes.

    Computes: (random_metric - uncertainty_metric) vs seed_size
    Positive values mean uncertainty is better.
    """
    seed_sizes = sorted(results.keys())
    advantages = []
    advantages_sem = []

    for seed_size in seed_sizes:
        if 'random' not in results[seed_size] or 'uncertainty' not in results[seed_size]:
            continue

        random_runs = results[seed_size]['random']['runs']
        uncertainty_runs = results[seed_size]['uncertainty']['runs']

        # Compute advantage for each random seed
        # We'll use the final metric value (convert to float to handle string values from JSON)
        random_final = [float(run[metric][-1]) for run in random_runs if run[metric]]
        uncertainty_final = [float(run[metric][-1]) for run in uncertainty_runs if run[metric]]

        if random_final and uncertainty_final:
            # For RMSE, lower is better, so advantage = random - uncertainty
            # For R2, higher is better, so advantage = uncertainty - random
            if 'rmse' in metric or 'mae' in metric:
                adv = np.mean(random_final) - np.mean(uncertainty_final)
            else:  # r2
                adv = np.mean(uncertainty_final) - np.mean(random_final)

            # SEM of difference
            random_sem = np.std(random_final) / np.sqrt(len(random_final))
            uncertainty_sem = np.std(uncertainty_final) / np.sqrt(len(uncertainty_final))
            adv_sem = np.sqrt(random_sem**2 + uncertainty_sem**2)

            advantages.append(adv)
            advantages_sem.append(adv_sem)
        else:
            advantages.append(np.nan)
            advantages_sem.append(0)

    # Plot
    fig, ax = plt.subplots(figsize=(8, 6))

    ax.errorbar(
        seed_sizes,
        advantages,
        yerr=advantages_sem,
        marker='o',
        linewidth=2,
        markersize=8,
        capsize=5,
        color='darkgreen'
    )

    ax.axhline(0, color='gray', linestyle='--', linewidth=1)
    ax.set_xlabel('Initial Seed Size', fontweight='bold')

    if 'rmse' in metric or 'mae' in metric:
        ax.set_ylabel(f'Advantage (Random - Uncertainty)\n{metric.upper()}', fontweight='bold')
    else:
        ax.set_ylabel(f'Advantage (Uncertainty - Random)\n{metric.upper()}', fontweight='bold')

    ax.set_title('Cold Start Advantage of Uncertainty Sampling', fontweight='bold')
    ax.set_xscale('log')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    output_path = output_dir / f'cold_start_advantage_{metric}.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved: {output_path}")
    plt.close()


def compute_statistical_tests(results: Dict) -> pd.DataFrame:
    """
    Compute paired t-tests between strategies at each seed size.

    For each seed size, compare final test RMSE between:
    - random vs uncertainty
    - random vs spatial
    - random vs hybrid
    - uncertainty vs hybrid

    Returns:
        DataFrame with test results
    """
    test_results = []

    for seed_size in sorted(results.keys()):
        strategies = list(results[seed_size].keys())

        # Pairs to compare
        pairs = [
            ('random', 'uncertainty'),
            ('random', 'spatial'),
            ('random', 'hybrid'),
            ('uncertainty', 'hybrid')
        ]

        for strategy1, strategy2 in pairs:
            if strategy1 not in strategies or strategy2 not in strategies:
                continue

            # Get final RMSE for each run
            runs1 = results[seed_size][strategy1]['runs']
            runs2 = results[seed_size][strategy2]['runs']

            # Use final test_rmse (convert to float to handle string values from JSON)
            values1 = [float(run['test_rmse'][-1]) for run in runs1 if run['test_rmse']]
            values2 = [float(run['test_rmse'][-1]) for run in runs2 if run['test_rmse']]

            if not values1 or not values2:
                continue

            # Paired t-test (assumes same random seeds used)
            n = min(len(values1), len(values2))
            values1 = values1[:n]
            values2 = values2[:n]

            t_stat, p_value = stats.ttest_rel(values1, values2)

            # Effect size (Cohen's d for paired samples)
            diff = np.array(values1) - np.array(values2)
            cohens_d = np.mean(diff) / np.std(diff)

            test_results.append({
                'seed_size': seed_size,
                'strategy1': strategy1,
                'strategy2': strategy2,
                'mean1': np.mean(values1),
                'mean2': np.mean(values2),
                'mean_diff': np.mean(values1) - np.mean(values2),
                't_statistic': t_stat,
                'p_value': p_value,
                'cohens_d': cohens_d,
                'n_pairs': n,
                'significant': p_value < 0.05
            })

    return pd.DataFrame(test_results)


def generate_summary_table(results: Dict) -> pd.DataFrame:
    """
    Generate summary table with final metrics for each configuration.

    Returns:
        DataFrame with mean ± SEM for final metrics
    """
    summary_data = []

    for seed_size in sorted(results.keys()):
        for strategy, data in results[seed_size].items():
            runs = data['runs']

            if not runs:
                continue

            # Extract final metrics (convert to float to handle string values from JSON)
            final_rmse = [float(run['test_rmse'][-1]) for run in runs if run['test_rmse']]
            final_mae = [float(run['test_mae'][-1]) for run in runs if run['test_mae']]
            final_r2 = [float(run['test_r2'][-1]) for run in runs if run['test_r2']]
            final_n_train = [int(run['n_train'][-1]) for run in runs if run['n_train']]

            summary_data.append({
                'seed_size': seed_size,
                'strategy': strategy,
                'n_runs': len(runs),
                'final_n_train': f"{np.mean(final_n_train):.0f} ± {np.std(final_n_train):.0f}",
                'test_rmse': f"{np.mean(final_rmse):.4f} ± {np.std(final_rmse)/np.sqrt(len(final_rmse)):.4f}",
                'test_mae': f"{np.mean(final_mae):.4f} ± {np.std(final_mae)/np.sqrt(len(final_mae)):.4f}",
                'test_r2': f"{np.mean(final_r2):.4f} ± {np.std(final_r2)/np.sqrt(len(final_r2)):.4f}",
            })

    return pd.DataFrame(summary_data)


def main():
    parser = argparse.ArgumentParser(description="Analyze sweep experiment results")
    parser.add_argument('sweep_dir', type=str,
                        help='Directory containing sweep results')
    parser.add_argument('--output-dir', type=str, default=None,
                        help='Output directory for plots (default: sweep_dir/analysis)')
    parser.add_argument('--target-rmse', type=float, default=0.5,
                        help='Target RMSE for sample efficiency (default: 0.5)')

    args = parser.parse_args()

    sweep_dir = Path(args.sweep_dir)
    if not sweep_dir.exists():
        print(f"Error: Sweep directory not found: {sweep_dir}")
        return

    # Output directory
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        output_dir = sweep_dir / 'analysis'

    output_dir.mkdir(parents=True, exist_ok=True)

    print("Loading sweep results...")
    results = load_sweep_results(sweep_dir)

    if not results:
        print("No results found!")
        return

    # Print summary
    print(f"\nLoaded results:")
    for seed_size in sorted(results.keys()):
        print(f"\n  Seed size {seed_size}:")
        for strategy, data in results[seed_size].items():
            print(f"    {strategy}: {len(data['runs'])} runs")

    # Generate summary table
    print("\nGenerating summary table...")
    summary_df = generate_summary_table(results)
    summary_df.to_csv(output_dir / 'summary_table.csv', index=False)
    print(f"Saved: {output_dir / 'summary_table.csv'}")
    print("\n" + summary_df.to_string(index=False))

    # Plot learning curves
    print("\nGenerating learning curve plots...")
    for metric, label in [
        ('test_rmse', 'Test RMSE (log space)'),
        ('test_mae', 'Test MAE (log space)'),
        ('test_r2', 'Test R²')
    ]:
        plot_learning_curves(results, output_dir, metric, label)

    # Sample efficiency
    print("\nComputing sample efficiency...")
    efficiency_df = compute_sample_efficiency(results, args.target_rmse)
    if not efficiency_df.empty:
        efficiency_df.to_csv(output_dir / 'sample_efficiency.csv', index=False)
        print(f"Saved: {output_dir / 'sample_efficiency.csv'}")
        print("\n" + efficiency_df.to_string(index=False))

        plot_sample_efficiency(efficiency_df, output_dir)

    # Cold start advantage
    print("\nPlotting cold start advantage...")
    for metric in ['test_rmse', 'test_r2']:
        plot_cold_start_advantage(results, output_dir, metric)

    # Statistical tests
    print("\nComputing statistical tests...")
    stats_df = compute_statistical_tests(results)
    if not stats_df.empty:
        stats_df.to_csv(output_dir / 'statistical_tests.csv', index=False)
        print(f"Saved: {output_dir / 'statistical_tests.csv'}")
        print("\nSignificant differences (p < 0.05):")
        sig_df = stats_df[stats_df['significant']].copy()
        if not sig_df.empty:
            print(sig_df[['seed_size', 'strategy1', 'strategy2', 'mean_diff', 'p_value', 'cohens_d']].to_string(index=False))
        else:
            print("  None found")

    print(f"\n{'='*60}")
    print("Analysis Complete!")
    print(f"Results saved to: {output_dir}")
    print(f"{'='*60}")


if __name__ == '__main__':
    main()
