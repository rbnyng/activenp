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
from active_learning import get_sampler, SimpleActiveLearningLoop
from active_learning.rf_loop import RFActiveLearningLoop
from models.neural_process import GEDINeuralProcess
from models.random_forest_baseline import RandomForestBaseline
from utils.config import save_config
from utils.spatial import split_data_spatial_block


def query_and_prepare_data(
    bbox: tuple,
    year: int = 2024,
    cache_dir: Path = Path("./cache"),
    sample_limit: int = None,
    agbd_max: float = 500.0
) -> pd.DataFrame:
    """
    Query GEDI data and extract embeddings for the specified region.

    Args:
        bbox: (lon_min, lat_min, lon_max, lat_max)
        year: Year for GeoTessera embeddings
        cache_dir: Directory for caching
        sample_limit: Maximum number of samples to keep (for testing)
        agbd_max: Maximum AGBD threshold to filter outliers (default: 500 Mg/ha)

    Returns:
        DataFrame with columns: latitude, longitude, agbd, embedding_patch, tile_id
    """
    print(f"\n{'='*60}")
    print("Step 1: Querying GEDI data")
    print(f"{'='*60}")
    print(f"Region: {bbox}")
    print(f"Year: {year}")

    start_year = 2022
    end_year = 2022

    querier = GEDIQuerier(cache_dir=cache_dir / "gedi")
    gedi_df = querier.query_bbox(
        bbox=bbox,
        start_time=f"{start_year}-01-01",
        end_time=f"{end_year}-12-31"
    )

    print(f"Found {len(gedi_df)} GEDI shots ({start_year}-{end_year})")

    # Print raw AGBD statistics
    print(f"\nRaw AGBD statistics:")
    print(f"  Mean: {gedi_df['agbd'].mean():.1f} Mg/ha")
    print(f"  Std:  {gedi_df['agbd'].std():.1f} Mg/ha")
    print(f"  Min:  {gedi_df['agbd'].min():.1f} Mg/ha")
    print(f"  Max:  {gedi_df['agbd'].max():.1f} Mg/ha")

    # Filter outliers
    n_before = len(gedi_df)
    gedi_df = gedi_df[gedi_df['agbd'] <= agbd_max].copy()
    n_after = len(gedi_df)

    if n_before > n_after:
        print(f"\nFiltered {n_before - n_after} shots with AGBD > {agbd_max} Mg/ha")
        print(f"\nFiltered AGBD statistics:")
        print(f"  Mean: {gedi_df['agbd'].mean():.1f} Mg/ha")
        print(f"  Std:  {gedi_df['agbd'].std():.1f} Mg/ha")
        print(f"  Min:  {gedi_df['agbd'].min():.1f} Mg/ha")
        print(f"  Max:  {gedi_df['agbd'].max():.1f} Mg/ha")

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
        patch_size=3,
        cache_dir=cache_dir / "geotessera"
    )

    gedi_df = extractor.extract_patches_batch(gedi_df)

    # Filter out failed extractions (None embeddings)
    n_before = len(gedi_df)
    gedi_df = gedi_df[gedi_df['embedding_patch'].notna()].copy()
    n_after = len(gedi_df)

    if n_before > n_after:
        print(f"Filtered out {n_before - n_after} shots with failed embedding extraction")

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
    n_pool: int = None,
    seed: int = 42,
    spatial_blocking: bool = False,
    test_fraction: float = 0.2,
    n_blocks_lon: int = 4,
    n_blocks_lat: int = 4
) -> tuple:
    """
    Split data into seed (initial training), pool, and test sets.

    Args:
        df: Full dataset
        n_seed: Size of initial seed set
        n_pool: Size of pool for active learning (None = use half of remaining)
        seed: Random seed
        spatial_blocking: If True, use spatially-blocked test set (gold standard)
        test_fraction: Fraction of data for spatially-blocked test set
        n_blocks_lon: Number of spatial blocks along longitude
        n_blocks_lat: Number of spatial blocks along latitude

    Returns:
        (seed_df, pool_df, test_df)

    Note: test_df contains ALL remaining data to evaluate how well the
    sampling policy covers the entire AOI. If spatial_blocking=True,
    the test set is spatially separated from training data.
    """
    print(f"\n{'='*60}")
    print("Step 3: Splitting data")
    print(f"{'='*60}")

    if spatial_blocking:
        print("Using spatially-blocked test set (gold standard for geospatial evaluation)")
        return split_data_spatial_block(
            df=df,
            n_seed=n_seed,
            n_pool=n_pool,
            test_fraction=test_fraction,
            n_blocks_lon=n_blocks_lon,
            n_blocks_lat=n_blocks_lat,
            seed=seed
        )
    else:
        print("Using random test set")
        np.random.seed(seed)
        df = df.sample(frac=1, random_state=seed).reset_index(drop=True)

        # Split: seed + pool + test (everything else)
        seed_df = df.iloc[:n_seed].copy()
        remaining = df.iloc[n_seed:].copy()

        # Pool size: either specified or half of remaining data
        if n_pool is None:
            n_pool = len(remaining) // 2

        pool_df = remaining.iloc[:n_pool].copy()
        test_df = remaining.iloc[n_pool:].copy()  # Everything else as test

        print(f"Seed (initial train): {len(seed_df)}")
        print(f"Pool: {len(pool_df)}")
        print(f"Test (full AOI coverage): {len(test_df)}")

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
    output_dir: Path = None,
    model_type: str = 'anp'
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
        model_type: 'anp' or 'rf' (Random Forest baseline)

    Returns:
        History dictionary
    """
    print(f"\n{'='*60}")
    print(f"Running Active Learning: {model_type.upper()} - {strategy.upper()}")
    print(f"{'='*60}")

    # Reset random seeds for consistent initialization across strategies
    torch.manual_seed(config['seed'])
    torch.cuda.manual_seed_all(config['seed'])
    np.random.seed(config['seed'])

    # Get sampler
    if strategy == 'random':
        sampler = get_sampler('random', seed=config['seed'])
    elif strategy == 'uncertainty':
        sampler = get_sampler('uncertainty')
    elif strategy == 'spatial':
        sampler = get_sampler('spatial', method='maxmin')
    elif strategy == 'hybrid':
        sampler = get_sampler('hybrid', uncertainty_percentile=0.75, spatial_method='maxmin')
    elif strategy == 'hybrid_product':
        sampler = get_sampler('hybrid_product', distance_weight=1.0)
    else:
        raise ValueError(f"Unknown strategy: {strategy}")

    if model_type == 'rf':
        # Random Forest baseline
        model = RandomForestBaseline(
            n_estimators=100,
            max_depth=10,
            random_state=config['seed']
        )

        al_loop = RFActiveLearningLoop(
            model=model,
            sampler=sampler,
            verbose=True
        )
    else:
        # ANP model
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

        al_loop = SimpleActiveLearningLoop(
            model=model,
            optimizer=optimizer,
            device=device,
            sampler=sampler,
            batch_size=config['batch_size'],
            epochs_per_iteration=config['epochs_per_iteration'],
            kl_weight=config['kl_weight'],
            global_bounds=config['global_bounds'],
            context_ratio=0.5,  # Use 50% as context, 50% as targets
            verbose=True
        )

    # Run
    save_dir = output_dir / f"{model_type}_{strategy}" if output_dir else None
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
        histories: Dict mapping (model_type, strategy) to history dict
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

    # Colors for ANP strategies
    colors = {
        'random': 'gray',
        'uncertainty': 'red',
        'spatial': 'blue',
        'hybrid': 'green',
        'hybrid_product': 'purple'
    }

    for metric_key, metric_name, ax in metrics:
        for (model_type, strategy_name), history in histories.items():
            # Determine color and style
            if model_type == 'rf':
                color = 'orange'
                linestyle = '--'
                label = f"RF-{strategy_name.capitalize()}"
            else:
                color = colors.get(strategy_name, 'black')
                linestyle = '-'
                label = f"ANP-{strategy_name.capitalize()}"

            ax.plot(
                history['n_train'],
                history[metric_key],
                marker='o',
                label=label,
                color=color,
                linestyle=linestyle,
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


def convert_to_native_types(obj):
    """
    Recursively convert numpy/torch types to Python native types for JSON serialization.

    Args:
        obj: Object to convert (can be dict, list, numpy array, numpy scalar, etc.)

    Returns:
        Object with all numpy/torch types converted to Python native types
    """
    if isinstance(obj, dict):
        return {key: convert_to_native_types(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [convert_to_native_types(item) for item in obj]
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, (np.integer, np.floating)):
        return obj.item()
    elif isinstance(obj, np.bool_):
        return bool(obj)
    elif hasattr(obj, 'tolist'):  # torch tensors and similar
        return obj.tolist()
    else:
        return obj


def save_results(
    histories: dict,
    config: dict,
    output_dir: Path
):
    """Save results to JSON."""
    # Convert numpy/torch types to Python native types
    histories_converted = convert_to_native_types(histories)
    config_converted = convert_to_native_types(config)

    results = {
        'config': config_converted,
        'histories': histories_converted,
        'timestamp': datetime.now().isoformat()
    }

    output_path = output_dir / 'results.json'
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"Saved results to: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Run active learning experiment")
    parser.add_argument('--bbox', type=float, nargs=4, default=[-70.0, 44.0, -69.0, 45.0],
                        help='Bounding box: lon_min lat_min lon_max lat_max (default: Maine)')
    parser.add_argument('--year', type=int, default=2022,
                        help='Year for GeoTessera embeddings (default: 2022)')
    parser.add_argument('--agbd-max', type=float, default=500.0,
                        help='Maximum AGBD threshold to filter outliers (default: 500 Mg/ha)')
    parser.add_argument('--n-seed', type=int, default=100,
                        help='Initial seed size')
    parser.add_argument('--n-pool', type=int, default=None,
                        help='Pool size for active learning (default: half of remaining data)')
    parser.add_argument('--n-iterations', type=int, default=15,
                        help='Number of AL iterations')
    parser.add_argument('--samples-per-iter', type=int, default=10,
                        help='Samples to acquire per iteration')
    parser.add_argument('--strategies', type=str, nargs='+',
                        default=['random', 'uncertainty', 'spatial', 'hybrid', 'hybrid_product'],
                        help='Sampling strategies to compare')
    parser.add_argument('--model-types', type=str, nargs='+',
                        default=['anp'],
                        help='Model types to compare: anp, rf (default: anp)')
    parser.add_argument('--spatial-blocking', action='store_true',
                        help='Use spatially-blocked test set (gold standard for geospatial evaluation)')
    parser.add_argument('--test-fraction', type=float, default=0.2,
                        help='Fraction of data for spatially-blocked test set (default: 0.2)')
    parser.add_argument('--n-blocks-lon', type=int, default=4,
                        help='Number of spatial blocks along longitude (default: 4)')
    parser.add_argument('--n-blocks-lat', type=int, default=4,
                        help='Number of spatial blocks along latitude (default: 4)')
    parser.add_argument('--output-dir', type=str, default='./results/active_learning',
                        help='Output directory')
    parser.add_argument('--cache-dir', type=str, default='./cache',
                        help='Cache directory')
    parser.add_argument('--sample-limit', type=int, default=None,
                        help='Limit total samples (for testing)')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu',
                        help='Device to use')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for data splitting (default: 42)')

    args = parser.parse_args()

    # Setup
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    cache_dir = Path(args.cache_dir)

    device = torch.device(args.device)
    print(f"Using device: {device}")

    # Set random seeds
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    np.random.seed(args.seed)

    # Model configuration
    config = {
        'patch_size': 3,
        'embedding_feature_dim': 1024,
        'context_repr_dim': 256,
        'hidden_dim': 512,
        'latent_dim': 256,
        'architecture_mode': 'anp',
        'num_attention_heads': 16,
        'learning_rate': 5e-4,
        'batch_size': 32,
        'epochs_per_iteration': 30,
        'kl_weight': 0.1,
        'global_bounds': tuple(args.bbox),
        'bbox': args.bbox,
        'year': args.year,  # Track which year of embeddings was used
        'agbd_max': args.agbd_max,  # Maximum AGBD threshold
        'n_seed': args.n_seed,
        'n_pool': args.n_pool,
        'n_iterations': args.n_iterations,
        'samples_per_iteration': args.samples_per_iter,
        'seed': args.seed
    }

    # Save config
    save_config(config, output_dir / 'config.json')

    # Query and prepare data
    df = query_and_prepare_data(
        bbox=tuple(args.bbox),
        year=args.year,
        cache_dir=cache_dir,
        sample_limit=args.sample_limit,
        agbd_max=args.agbd_max
    )

    # Split data
    seed_df, pool_df, test_df = split_data(
        df,
        n_seed=args.n_seed,
        n_pool=args.n_pool,
        seed=args.seed,
        spatial_blocking=args.spatial_blocking,
        test_fraction=args.test_fraction,
        n_blocks_lon=args.n_blocks_lon,
        n_blocks_lat=args.n_blocks_lat
    )

    # Save splits
    seed_df.to_pickle(output_dir / 'seed_df.pkl')
    pool_df.to_pickle(output_dir / 'pool_df.pkl')
    test_df.to_pickle(output_dir / 'test_df.pkl')

    # Run active learning for each model type and strategy combination
    histories = {}
    for model_type in args.model_types:
        for strategy in args.strategies:
            print(f"\n{'='*80}")
            print(f"Running: {model_type.upper()} with {strategy.upper()} strategy")
            print(f"{'='*80}")

            history = run_active_learning(
                seed_df=seed_df,
                pool_df=pool_df,
                test_df=test_df,
                strategy=strategy,
                config=config,
                device=device,
                n_iterations=args.n_iterations,
                samples_per_iteration=args.samples_per_iter,
                output_dir=output_dir,
                model_type=model_type
            )
            histories[(model_type, strategy)] = history

    # Plot and save results
    plot_learning_curves(histories, output_dir)
    save_results(histories, config, output_dir)

    print(f"\n{'='*60}")
    print("Experiment Complete!")
    print(f"Results saved to: {output_dir}")
    print(f"{'='*60}\n")


if __name__ == '__main__':
    main()
