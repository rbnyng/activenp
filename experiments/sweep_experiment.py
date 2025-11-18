"""
Seed sweep experiment harness for active learning.

This script runs multiple active learning experiments with different random seeds
to quantify variance and statistical significance of results.

Usage:
    python experiments/sweep_experiment.py --n-seeds 5 [other args...]
"""

import sys
from pathlib import Path

# Add project root to path for imports
project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

import argparse
import json
import subprocess
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
from typing import Dict, List
import pickle

from scipy import stats


def run_single_experiment(
    seed: int,
    output_dir: Path,
    **kwargs
) -> Dict:
    """
    Run a single experiment with given seed.

    Args:
        seed: Random seed
        output_dir: Output directory for this seed's results
        **kwargs: Additional arguments to pass to experiment

    Returns:
        Dictionary with results path and metadata
    """
    print(f"\n{'='*80}")
    print(f"Running experiment with seed {seed}")
    print(f"{'='*80}\n")

    # Build command
    cmd = [
        'python',
        'experiments/active_learning_experiment.py',
        '--seed', str(seed),
        '--output-dir', str(output_dir)
    ]

    # Add other arguments
    for key, value in kwargs.items():
        if value is None:
            continue

        # Convert key from underscore to dash
        arg_name = f"--{key.replace('_', '-')}"

        # Handle list arguments
        if isinstance(value, list):
            cmd.append(arg_name)
            cmd.extend(str(v) for v in value)
        else:
            cmd.extend([arg_name, str(value)])

    # Run experiment
    try:
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        print(result.stdout)
        if result.stderr:
            print("STDERR:", result.stderr)
    except subprocess.CalledProcessError as e:
        print(f"Error running experiment with seed {seed}:")
        print(e.stdout)
        print(e.stderr)
        raise

    # Load results
    results_path = output_dir / 'results.json'
    with open(results_path, 'r') as f:
        results = json.load(f)

    return {
        'seed': seed,
        'results_path': str(results_path),
        'histories': results['histories']
    }


def aggregate_results(
    seed_results: List[Dict],
    strategies: List[str]
) -> Dict:
    """
    Aggregate results across seeds.

    Args:
        seed_results: List of results dicts from each seed
        strategies: List of strategy names

    Returns:
        Aggregated statistics
    """
    print(f"\n{'='*80}")
    print("Aggregating results across seeds")
    print(f"{'='*80}\n")

    aggregated = {}

    for strategy in strategies:
        # Collect all histories for this strategy
        all_histories = []
        for seed_result in seed_results:
            if strategy in seed_result['histories']:
                all_histories.append(seed_result['histories'][strategy])

        if not all_histories:
            continue

        # Get metrics to aggregate
        metrics = ['test_rmse', 'test_mae', 'test_r2', 'train_loss']

        # Aggregate each metric
        strategy_stats = {}
        for metric in metrics:
            # Extract metric values across seeds
            metric_values = []
            for hist in all_histories:
                if metric in hist:
                    metric_values.append(hist[metric])

            if metric_values:
                # Convert to array (seeds x iterations)
                metric_array = np.array(metric_values)

                # Compute statistics
                strategy_stats[metric] = {
                    'mean': metric_array.mean(axis=0).tolist(),
                    'std': metric_array.std(axis=0).tolist(),
                    'min': metric_array.min(axis=0).tolist(),
                    'max': metric_array.max(axis=0).tolist(),
                    'median': np.median(metric_array, axis=0).tolist(),
                    'values': metric_array.tolist()  # All raw values
                }

        # Get n_train (should be same across all seeds)
        strategy_stats['n_train'] = all_histories[0]['n_train']

        aggregated[strategy] = strategy_stats

    return aggregated


def plot_learning_curves_with_confidence(
    aggregated: Dict,
    output_dir: Path,
    n_seeds: int
):
    """
    Plot learning curves with confidence bands.

    Args:
        aggregated: Aggregated statistics from aggregate_results
        output_dir: Directory to save plots
        n_seeds: Number of seeds used
    """
    print(f"\n{'='*80}")
    print("Generating plots with confidence bands")
    print(f"{'='*80}\n")

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    metrics = [
        ('test_rmse', 'Test RMSE (log space)', axes[0, 0], False),
        ('test_mae', 'Test MAE (log space)', axes[0, 1], False),
        ('test_r2', 'Test R²', axes[1, 0], True),  # Higher is better
        ('train_loss', 'Training Loss', axes[1, 1], False)
    ]

    colors = {
        'random': 'gray',
        'uncertainty': 'red',
        'spatial': 'blue',
        'hybrid': 'green'
    }

    for metric_key, metric_name, ax, higher_better in metrics:
        for strategy_name, stats in aggregated.items():
            if metric_key not in stats:
                continue

            n_train = stats['n_train']
            mean = np.array(stats[metric_key]['mean'])
            std = np.array(stats[metric_key]['std'])

            color = colors.get(strategy_name, 'black')

            # Plot mean line
            line = ax.plot(
                n_train,
                mean,
                marker='o',
                label=f"{strategy_name.capitalize()} (n={n_seeds})",
                color=color,
                linewidth=2,
                markersize=6
            )[0]

            # Plot confidence band (mean ± 1 std)
            ax.fill_between(
                n_train,
                mean - std,
                mean + std,
                alpha=0.2,
                color=color
            )

        ax.set_xlabel('Number of Training Samples', fontweight='bold')
        ax.set_ylabel(metric_name, fontweight='bold')
        ax.set_title(metric_name, fontweight='bold')
        ax.legend(loc='best')
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    output_path = output_dir / 'learning_curves_with_confidence.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved learning curves to: {output_path}")
    plt.close()


def create_summary_table(
    aggregated: Dict,
    output_dir: Path
):
    """
    Create summary table with final performance metrics.

    Args:
        aggregated: Aggregated statistics
        output_dir: Directory to save table
    """
    print(f"\n{'='*80}")
    print("Creating summary table")
    print(f"{'='*80}\n")

    rows = []

    for strategy_name, stats in aggregated.items():
        row = {'strategy': strategy_name}

        # Get final values (last iteration)
        for metric in ['test_rmse', 'test_mae', 'test_r2']:
            if metric in stats:
                mean_values = stats[metric]['mean']
                std_values = stats[metric]['std']

                final_mean = mean_values[-1]
                final_std = std_values[-1]

                row[f'{metric}_mean'] = final_mean
                row[f'{metric}_std'] = final_std
                row[f'{metric}_formatted'] = f"{final_mean:.4f} ± {final_std:.4f}"

        rows.append(row)

    df = pd.DataFrame(rows)

    # Save as CSV
    csv_path = output_dir / 'summary_table.csv'
    df.to_csv(csv_path, index=False)
    print(f"Saved summary table to: {csv_path}")

    # Print to console
    print("\nFinal Performance (Mean ± Std):")
    print("="*80)
    for _, row in df.iterrows():
        print(f"\n{row['strategy'].upper()}:")
        if 'test_rmse_formatted' in row:
            print(f"  RMSE: {row['test_rmse_formatted']}")
        if 'test_mae_formatted' in row:
            print(f"  MAE:  {row['test_mae_formatted']}")
        if 'test_r2_formatted' in row:
            print(f"  R²:   {row['test_r2_formatted']}")
    print("="*80)


def statistical_tests(
    aggregated: Dict,
    output_dir: Path
):
    """
    Perform statistical significance tests.

    Args:
        aggregated: Aggregated statistics
        output_dir: Directory to save results
    """
    print(f"\n{'='*80}")
    print("Running statistical tests")
    print(f"{'='*80}\n")

    # Use Wilcoxon signed-rank test for pairwise comparisons
    # Focus on final test_rmse

    strategies = list(aggregated.keys())
    if len(strategies) < 2:
        print("Need at least 2 strategies for statistical tests")
        return

    results = {}

    for metric in ['test_rmse', 'test_mae', 'test_r2']:
        metric_results = {}

        # Get final values for each strategy
        final_values = {}
        for strategy in strategies:
            if metric in aggregated[strategy]:
                values = aggregated[strategy][metric]['values']
                # Get final iteration values across all seeds
                final_values[strategy] = [seed_vals[-1] for seed_vals in values]

        # Pairwise comparisons
        for i, strategy1 in enumerate(strategies):
            if strategy1 not in final_values:
                continue

            for strategy2 in strategies[i+1:]:
                if strategy2 not in final_values:
                    continue

                vals1 = final_values[strategy1]
                vals2 = final_values[strategy2]

                # Wilcoxon signed-rank test (paired test)
                if len(vals1) == len(vals2) and len(vals1) > 0:
                    try:
                        stat, p_value = stats.wilcoxon(vals1, vals2)

                        comparison_key = f"{strategy1}_vs_{strategy2}"
                        metric_results[comparison_key] = {
                            'statistic': float(stat),
                            'p_value': float(p_value),
                            'significant_05': p_value < 0.05,
                            'significant_01': p_value < 0.01,
                            f'{strategy1}_mean': float(np.mean(vals1)),
                            f'{strategy2}_mean': float(np.mean(vals2)),
                            'winner': strategy1 if np.mean(vals1) < np.mean(vals2) else strategy2
                        }
                    except Exception as e:
                        print(f"Could not perform test for {strategy1} vs {strategy2}: {e}")

        results[metric] = metric_results

    # Save results
    output_path = output_dir / 'statistical_tests.json'
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"Saved statistical tests to: {output_path}")

    # Print significant results
    print("\nStatistical Significance (p < 0.05):")
    print("="*80)
    for metric, metric_results in results.items():
        print(f"\n{metric.upper()}:")
        for comparison, result in metric_results.items():
            if result['significant_05']:
                print(f"  {comparison}: p={result['p_value']:.4f} "
                      f"(winner: {result['winner']})")
    print("="*80)


def main():
    parser = argparse.ArgumentParser(
        description="Run seed sweep for active learning experiments"
    )

    # Sweep-specific arguments
    parser.add_argument('--n-seeds', type=int, default=5,
                        help='Number of random seeds to run (default: 5)')
    parser.add_argument('--start-seed', type=int, default=42,
                        help='Starting seed value (default: 42)')

    # Experiment arguments (pass through to active_learning_experiment.py)
    parser.add_argument('--bbox', type=float, nargs=4, default=[-70.0, 44.0, -69.0, 45.0],
                        help='Bounding box: lon_min lat_min lon_max lat_max')
    parser.add_argument('--year', type=int, default=2022,
                        help='Year for GeoTessera embeddings')
    parser.add_argument('--agbd-max', type=float, default=500.0,
                        help='Maximum AGBD threshold')
    parser.add_argument('--n-seed', type=int, default=100,
                        help='Initial seed size')
    parser.add_argument('--n-pool', type=int, default=None,
                        help='Pool size for active learning')
    parser.add_argument('--n-iterations', type=int, default=15,
                        help='Number of AL iterations')
    parser.add_argument('--samples-per-iter', type=int, default=10,
                        help='Samples to acquire per iteration')
    parser.add_argument('--strategies', type=str, nargs='+',
                        default=['random', 'uncertainty', 'spatial', 'hybrid'],
                        help='Sampling strategies to compare')
    parser.add_argument('--output-dir', type=str, default=None,
                        help='Output directory (default: ./results/sweep_TIMESTAMP)')
    parser.add_argument('--cache-dir', type=str, default='./cache',
                        help='Cache directory')
    parser.add_argument('--sample-limit', type=int, default=None,
                        help='Limit total samples (for testing)')
    parser.add_argument('--device', type=str, default=None,
                        help='Device to use')

    args = parser.parse_args()

    # Setup output directory
    if args.output_dir is None:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        output_dir = Path(f'./results/sweep_{timestamp}')
    else:
        output_dir = Path(args.output_dir)

    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*80}")
    print(f"Seed Sweep Experiment")
    print(f"{'='*80}")
    print(f"Number of seeds: {args.n_seeds}")
    print(f"Output directory: {output_dir}")
    print(f"Strategies: {args.strategies}")
    print(f"{'='*80}\n")

    # Save sweep configuration
    sweep_config = {
        'n_seeds': args.n_seeds,
        'start_seed': args.start_seed,
        'seeds': list(range(args.start_seed, args.start_seed + args.n_seeds)),
        'timestamp': datetime.now().isoformat(),
        'experiment_args': {
            'bbox': args.bbox,
            'year': args.year,
            'agbd_max': args.agbd_max,
            'n_seed': args.n_seed,
            'n_pool': args.n_pool,
            'n_iterations': args.n_iterations,
            'samples_per_iter': args.samples_per_iter,
            'strategies': args.strategies,
            'cache_dir': args.cache_dir,
            'sample_limit': args.sample_limit,
            'device': args.device
        }
    }

    with open(output_dir / 'sweep_config.json', 'w') as f:
        json.dump(sweep_config, f, indent=2)

    # Run experiments for each seed
    seed_results = []
    for i in range(args.n_seeds):
        seed = args.start_seed + i
        seed_output_dir = output_dir / f'seed_{seed}'

        # Prepare experiment arguments
        exp_kwargs = {
            'bbox': args.bbox,
            'year': args.year,
            'agbd_max': args.agbd_max,
            'n_seed': args.n_seed,
            'n_pool': args.n_pool,
            'n_iterations': args.n_iterations,
            'samples_per_iter': args.samples_per_iter,
            'strategies': args.strategies,
            'cache_dir': args.cache_dir,
            'sample_limit': args.sample_limit,
            'device': args.device
        }

        # Run experiment
        result = run_single_experiment(seed, seed_output_dir, **exp_kwargs)
        seed_results.append(result)

    # Aggregate results
    aggregated = aggregate_results(seed_results, args.strategies)

    # Save aggregated results
    with open(output_dir / 'aggregated_results.json', 'w') as f:
        json.dump(aggregated, f, indent=2)

    # Save raw seed results for later analysis
    with open(output_dir / 'all_seed_results.pkl', 'wb') as f:
        pickle.dump(seed_results, f)

    # Generate plots and tables
    plot_learning_curves_with_confidence(aggregated, output_dir, args.n_seeds)
    create_summary_table(aggregated, output_dir)
    statistical_tests(aggregated, output_dir)

    print(f"\n{'='*80}")
    print("Seed Sweep Complete!")
    print(f"Results saved to: {output_dir}")
    print(f"{'='*80}\n")


if __name__ == '__main__':
    main()
