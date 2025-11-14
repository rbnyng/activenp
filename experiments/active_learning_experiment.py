"""
Active learning experiment for GEDI Neural Process.

This script runs the full active learning experiment:
1. Query GEDI data from specified region
2. Extract GeoTessera embeddings
3. Split into initial seed, pool, and test sets
4. Run active learning with different strategies
5. Compare learning curves
6. Save results and plots
"""

import sys
from pathlib import Path

# Add project root to path for imports
project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

import argparse
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import torch
from datetime import datetime

from data.gedi import GEDIQuerier
from data.embeddings import EmbeddingExtractor
from active_learning import get_sampler, ActiveLearningLoop
from models.neural_process import GEDINeuralProcess
from utils.config import save_config


def query_and_prepare_data(
    bbox: tuple,
    year: int = 2024,
    cache_dir: Path = Path("./cache"),
    sample_limit: int = None
) -> pd.DataFrame:
    """
    Query GEDI data and extract embeddings for the specified region.

    Args:
        bbox: (lon_min, lat_min, lon_max, lat_max)
        year: Year for GeoTessera embeddings
        cache_dir: Directory for caching
        sample_limit: Maximum number of samples to keep (for testing)

    Returns:
        DataFrame with columns: latitude, longitude, agbd, embedding_patch, tile_id
    """
    print(f"\n{'='*60}")
    print("Step 1: Querying GEDI data")
    print(f"{'='*60}")
    print(f"Region: {bbox}")
    print(f"Year: {year}")

    # Query GEDI data
    querier = GEDIQuerier(cache_dir=cache_dir / "gedi")
    gedi_df = querier.query_bbox(
        bbox=bbox,
        start_date="2019-01-01",
        end_date="2023-12-31"
    )

    print(f"Found {len(gedi_df)} GEDI shots")

    # Sample if limit specified
    if sample_limit and len(gedi_df) > sample_limit:
        gedi_df = gedi_df.sample(n=sample_limit, random_state=42)
        print(f"Sampled {len(gedi_df)} shots for testing")

    # Extract embeddings
    print(f"\n{'='*60}")
    print("Step 2: Extracting GeoTessera embeddings")
    print(f"{'='*60}")

    extractor = EmbeddingExtractor(
        year=year,
        cache_dir=cache_dir / "geotessera"
    )

    gedi_df = extractor.extract_patches_batch(gedi_df, patch_size=3)

    # Add tile_id for spatial grouping
    gedi_df['tile_id'] = (
        (gedi_df['longitude'] // 0.1).astype(int).astype(str) + "_" +
        (gedi_df['latitude'] // 0.1).astype(int).astype(str)
    )

    print(f"Extracted embeddings for {len(gedi_df)} shots")
    print(f"Number of tiles: {gedi_df['tile_id'].nunique()}")

    return gedi_df


def split_data(
    df: pd.DataFrame,
    n_seed: int = 100,
    n_test: int = 1000,
    seed: int = 42
) -> tuple:
    """
    Split data into seed (initial training), pool, and test sets.

    Args:
        df: Full dataset
        n_seed: Size of initial seed set
        n_test: Size of test set
        seed: Random seed

    Returns:
        (seed_df, pool_df, test_df)
    """
    print(f"\n{'='*60}")
    print("Step 3: Splitting data")
    print(f"{'='*60}")

    np.random.seed(seed)
    df = df.sample(frac=1, random_state=seed).reset_index(drop=True)

    # Split
    seed_df = df.iloc[:n_seed].copy()
    remaining = df.iloc[n_seed:].copy()

    test_df = remaining.iloc[:n_test].copy()
    pool_df = remaining.iloc[n_test:].copy()

    print(f"Seed (initial train): {len(seed_df)}")
    print(f"Pool: {len(pool_df)}")
    print(f"Test: {len(test_df)}")

    return seed_df, pool_df, test_df


def run_active_learning(
    seed_df: pd.DataFrame,
    pool_df: pd.DataFrame,
    test_df: pd.DataFrame,
    strategy: str,
    config: dict,
    device: torch.device,
    n_iterations: int = 15,
    samples_per_iteration: int = 10,
    output_dir: Path = None
) -> dict:
    """
    Run active learning with specified strategy.

    Args:
        seed_df: Initial training data
        pool_df: Pool of unlabeled samples
        test_df: Test set
        strategy: Sampling strategy name
        config: Model configuration
        device: Device to run on
        n_iterations: Number of AL iterations
        samples_per_iteration: Samples to acquire per iteration
        output_dir: Directory to save results

    Returns:
        History dictionary
    """
    print(f"\n{'='*60}")
    print(f"Running Active Learning: {strategy.upper()}")
    print(f"{'='*60}")

    # Initialize model
    model = GEDINeuralProcess(
        patch_size=config['patch_size'],
        embedding_channels=128,
        embedding_feature_dim=config['embedding_feature_dim'],
        context_repr_dim=config['context_repr_dim'],
        hidden_dim=config['hidden_dim'],
        latent_dim=config['latent_dim'],
        output_uncertainty=True,
        architecture_mode=config['architecture_mode'],
        num_attention_heads=config['num_attention_heads']
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=config['learning_rate'])

    # Get sampler
    if strategy == 'random':
        sampler = get_sampler('random', seed=42)
    elif strategy == 'uncertainty':
        sampler = get_sampler('uncertainty')
    elif strategy == 'spatial':
        sampler = get_sampler('spatial', method='maxmin')
    elif strategy == 'hybrid':
        sampler = get_sampler('hybrid', uncertainty_percentile=0.75, spatial_method='maxmin')
    else:
        raise ValueError(f"Unknown strategy: {strategy}")

    # Create active learning loop
    al_loop = ActiveLearningLoop(
        model=model,
        optimizer=optimizer,
        device=device,
        sampler=sampler,
        batch_size=config['batch_size'],
        epochs_per_iteration=config['epochs_per_iteration'],
        kl_weight=config['kl_weight'],
        global_bounds=config['global_bounds'],
        verbose=True
    )

    # Run
    save_dir = output_dir / strategy if output_dir else None
    history = al_loop.run(
        train_df=seed_df,
        pool_df=pool_df,
        test_df=test_df,
        n_iterations=n_iterations,
        samples_per_iteration=samples_per_iteration,
        save_dir=save_dir
    )

    return history


def plot_learning_curves(
    histories: dict,
    output_dir: Path
):
    """
    Plot learning curves comparing different strategies.

    Args:
        histories: Dict mapping strategy name to history dict
        output_dir: Directory to save plots
    """
    print(f"\n{'='*60}")
    print("Generating plots")
    print(f"{'='*60}")

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    metrics = [
        ('test_rmse', 'Test RMSE (log space)', axes[0, 0]),
        ('test_mae', 'Test MAE (log space)', axes[0, 1]),
        ('test_r2', 'Test R²', axes[1, 0]),
        ('train_loss', 'Training Loss', axes[1, 1])
    ]

    colors = {
        'random': 'gray',
        'uncertainty': 'red',
        'spatial': 'blue',
        'hybrid': 'green'
    }

    for metric_key, metric_name, ax in metrics:
        for strategy_name, history in histories.items():
            ax.plot(
                history['n_train'],
                history[metric_key],
                marker='o',
                label=strategy_name.capitalize(),
                color=colors.get(strategy_name, 'black'),
                linewidth=2,
                markersize=6
            )

        ax.set_xlabel('Number of Training Samples', fontweight='bold')
        ax.set_ylabel(metric_name, fontweight='bold')
        ax.set_title(metric_name, fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    output_path = output_dir / 'learning_curves.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved learning curves to: {output_path}")
    plt.close()


def save_results(
    histories: dict,
    config: dict,
    output_dir: Path
):
    """Save results to JSON."""
    results = {
        'config': config,
        'histories': histories,
        'timestamp': datetime.now().isoformat()
    }

    output_path = output_dir / 'results.json'
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2, default=str)

    print(f"Saved results to: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Run active learning experiment")
    parser.add_argument('--bbox', type=float, nargs=4, default=[-73.0, 2.9, -72.9, 3.0],
                        help='Bounding box: lon_min lat_min lon_max lat_max')
    parser.add_argument('--n-seed', type=int, default=100,
                        help='Initial seed size')
    parser.add_argument('--n-test', type=int, default=1000,
                        help='Test set size')
    parser.add_argument('--n-iterations', type=int, default=15,
                        help='Number of AL iterations')
    parser.add_argument('--samples-per-iter', type=int, default=10,
                        help='Samples to acquire per iteration')
    parser.add_argument('--strategies', type=str, nargs='+',
                        default=['random', 'uncertainty', 'spatial', 'hybrid'],
                        help='Sampling strategies to compare')
    parser.add_argument('--output-dir', type=str, default='./results/active_learning',
                        help='Output directory')
    parser.add_argument('--cache-dir', type=str, default='./cache',
                        help='Cache directory')
    parser.add_argument('--sample-limit', type=int, default=None,
                        help='Limit total samples (for testing)')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu',
                        help='Device to use')

    args = parser.parse_args()

    # Setup
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    cache_dir = Path(args.cache_dir)

    device = torch.device(args.device)
    print(f"Using device: {device}")

    # Model configuration
    config = {
        'patch_size': 3,
        'embedding_feature_dim': 128,
        'context_repr_dim': 128,
        'hidden_dim': 512,
        'latent_dim': 128,
        'architecture_mode': 'deterministic',
        'num_attention_heads': 4,
        'learning_rate': 1e-3,
        'batch_size': 32,
        'epochs_per_iteration': 50,
        'kl_weight': 0.1,
        'global_bounds': tuple(args.bbox),
        'bbox': args.bbox,
        'n_seed': args.n_seed,
        'n_test': args.n_test,
        'n_iterations': args.n_iterations,
        'samples_per_iteration': args.samples_per_iter
    }

    # Save config
    save_config(config, output_dir / 'config.json')

    # Query and prepare data
    df = query_and_prepare_data(
        bbox=tuple(args.bbox),
        year=2024,
        cache_dir=cache_dir,
        sample_limit=args.sample_limit
    )

    # Split data
    seed_df, pool_df, test_df = split_data(
        df,
        n_seed=args.n_seed,
        n_test=args.n_test,
        seed=42
    )

    # Save splits
    seed_df.to_pickle(output_dir / 'seed_df.pkl')
    pool_df.to_pickle(output_dir / 'pool_df.pkl')
    test_df.to_pickle(output_dir / 'test_df.pkl')

    # Run active learning for each strategy
    histories = {}
    for strategy in args.strategies:
        history = run_active_learning(
            seed_df=seed_df,
            pool_df=pool_df,
            test_df=test_df,
            strategy=strategy,
            config=config,
            device=device,
            n_iterations=args.n_iterations,
            samples_per_iteration=args.samples_per_iter,
            output_dir=output_dir
        )
        histories[strategy] = history

    # Plot and save results
    plot_learning_curves(histories, output_dir)
    save_results(histories, config, output_dir)

    print(f"\n{'='*60}")
    print("Experiment Complete!")
    print(f"Results saved to: {output_dir}")
    print(f"{'='*60}\n")


if __name__ == '__main__':
    main()
