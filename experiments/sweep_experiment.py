"""
Sweep experiment harness for active learning.

This script runs multiple experiments across different configurations:
- Different initial seed sizes (5, 10, 25, 100, 1000)
- Multiple random seeds for statistical significance (default: 10)
- All sampling strategies run together in each experiment
- Fixed number of iterations and samples per iteration

Each experiment (seed_size, random_seed) runs all strategies and saves
them together in a single results.json file.

Results are organized for easy aggregation and analysis.
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

import argparse
import json
import subprocess
from datetime import datetime
from typing import List, Dict
import itertools
from multiprocessing import Pool
from functools import partial


def run_single_experiment(
    config: Dict,
    base_cache_dir: Path,
    device: str = 'cuda',
    dry_run: bool = False
) -> Dict:
    """
    Run a single experiment configuration.

    Args:
        config: Configuration dict with keys:
            - seed_size: Initial seed size
            - random_seed: Random seed for data split
            - strategies: List of sampling strategies to run
            - bbox: Bounding box
            - year: Year for embeddings
            - agbd_max: Max AGBD threshold
            - n_iterations: Number of active learning iterations
            - samples_per_iter: Samples per iteration
            - output_dir: Output directory
            - cache_dir: Cache directory
        base_cache_dir: Base cache directory (shared across runs)
        device: Device to use
        dry_run: If True, print command instead of running

    Returns:
        Dict with status and output
    """
    seed_size = config['seed_size']
    random_seed = config['random_seed']
    strategies = config['strategies']
    output_dir = Path(config['output_dir'])

    # Use configured number of iterations
    samples_per_iter = config['samples_per_iter']
    n_iterations = config['n_iterations']

    # Build command
    cmd = [
        'python', 'experiments/active_learning_experiment.py',
        '--bbox', *[str(x) for x in config['bbox']],
        '--year', str(config['year']),
        '--agbd-max', str(config['agbd_max']),
        '--n-seed', str(seed_size),
        '--n-iterations', str(n_iterations),
        '--samples-per-iter', str(samples_per_iter),
        '--strategies', *strategies,  # Pass all strategies
        '--output-dir', str(output_dir),
        '--cache-dir', str(base_cache_dir),
        '--device', device
    ]

    # Only add --n-pool if specified (None means use default = half of remaining)
    if config['n_pool'] is not None:
        cmd.extend(['--n-pool', str(config['n_pool'])])

    if dry_run:
        print(f"[DRY RUN] Would run: {' '.join(cmd)}")
        return {'status': 'dry_run', 'config': config, 'command': ' '.join(cmd)}

    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)

    # Run experiment
    print(f"\n{'='*60}")
    print(f"Running: seed={seed_size}, random_seed={random_seed}, strategies={strategies}")
    print(f"Output: {output_dir}")
    print(f"{'='*60}")

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            check=True,
            cwd=project_root
        )

        return {
            'status': 'success',
            'config': config,
            'stdout': result.stdout,
            'stderr': result.stderr
        }

    except subprocess.CalledProcessError as e:
        print(f"ERROR: Experiment failed!")
        print(f"stdout: {e.stdout}")
        print(f"stderr: {e.stderr}")

        return {
            'status': 'failed',
            'config': config,
            'error': str(e),
            'stdout': e.stdout,
            'stderr': e.stderr
        }


def generate_experiment_configs(
    seed_sizes: List[int],
    n_random_seeds: int,
    strategies: List[str],
    bbox: List[float],
    year: int,
    agbd_max: float,
    n_iterations: int,
    samples_per_iter: int,
    n_pool: int,
    base_output_dir: Path
) -> List[Dict]:
    """
    Generate all experiment configurations for the sweep.

    Returns:
        List of config dicts
    """
    configs = []

    for seed_size, random_seed in itertools.product(
        seed_sizes,
        range(n_random_seeds)
    ):
        # Output directory structure: seed_X/run_Y
        # All strategies run together in one experiment
        output_dir = base_output_dir / f"seed_{seed_size}" / f"run_{random_seed}"

        config = {
            'seed_size': seed_size,
            'random_seed': random_seed,
            'strategies': strategies,  # Pass all strategies
            'bbox': bbox,
            'year': year,
            'agbd_max': agbd_max,
            'n_iterations': n_iterations,
            'samples_per_iter': samples_per_iter,
            'output_dir': str(output_dir),
            'n_pool': n_pool
        }

        configs.append(config)

    return configs


def run_sweep_sequential(
    configs: List[Dict],
    base_cache_dir: Path,
    device: str,
    dry_run: bool = False
) -> List[Dict]:
    """Run experiments sequentially."""
    results = []

    for i, config in enumerate(configs):
        print(f"\n[{i+1}/{len(configs)}] Running configuration...")
        result = run_single_experiment(config, base_cache_dir, device, dry_run)
        results.append(result)

        # Save intermediate results
        if not dry_run and (i + 1) % 10 == 0:
            save_sweep_results(results, Path(config['output_dir']).parent.parent.parent)

    return results


def run_sweep_parallel(
    configs: List[Dict],
    base_cache_dir: Path,
    device: str,
    n_workers: int,
    dry_run: bool = False
) -> List[Dict]:
    """
    Run experiments in parallel.

    Note: This uses multiprocessing, so each worker will use the same device.
    For multi-GPU, you'll want to modify this to assign devices to workers.
    """
    print(f"Running {len(configs)} experiments with {n_workers} parallel workers...")

    # Create partial function with fixed arguments
    run_func = partial(
        run_single_experiment,
        base_cache_dir=base_cache_dir,
        device=device,
        dry_run=dry_run
    )

    with Pool(n_workers) as pool:
        results = pool.map(run_func, configs)

    return results


def save_sweep_results(results: List[Dict], output_dir: Path):
    """Save sweep results summary."""
    output_dir.mkdir(parents=True, exist_ok=True)

    summary_path = output_dir / 'sweep_summary.json'

    summary = {
        'timestamp': datetime.now().isoformat(),
        'n_experiments': len(results),
        'n_success': sum(1 for r in results if r['status'] == 'success'),
        'n_failed': sum(1 for r in results if r['status'] == 'failed'),
        'results': [
            {
                'status': r['status'],
                'config': r['config'],
                'error': r.get('error', None)
            }
            for r in results
        ]
    }

    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)

    print(f"\nSaved sweep summary to: {summary_path}")
    print(f"Success: {summary['n_success']}/{summary['n_experiments']}")
    print(f"Failed: {summary['n_failed']}/{summary['n_experiments']}")


def main():
    parser = argparse.ArgumentParser(
        description="Run sweep of active learning experiments",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Full sweep with default settings (Maine)
  python experiments/sweep_experiment.py --output-dir ./results/sweep

  # Quick test with small configuration
  python experiments/sweep_experiment.py \\
      --seed-sizes 10 25 \\
      --n-random-seeds 3 \\
      --strategies random uncertainty \\
      --output-dir ./results/test_sweep \\
      --dry-run

  # Parallel execution with 4 workers
  python experiments/sweep_experiment.py \\
      --output-dir ./results/sweep \\
      --parallel \\
      --n-workers 4
"""
    )

    # Sweep configuration
    parser.add_argument('--seed-sizes', type=int, nargs='+',
                        default=[5, 10, 25, 100, 1000],
                        help='Initial seed sizes to sweep (default: 5 10 25 100 1000)')
    parser.add_argument('--n-random-seeds', type=int, default=10,
                        help='Number of random seeds per configuration (default: 10)')
    parser.add_argument('--strategies', type=str, nargs='+',
                        default=['random', 'uncertainty', 'spatial', 'hybrid'],
                        help='Sampling strategies to compare (default: all)')

    # Experiment parameters
    parser.add_argument('--bbox', type=float, nargs=4, default=[-70.0, 44.0, -69.0, 45.0],
                        help='Bounding box: lon_min lat_min lon_max lat_max (default: Maine)')
    parser.add_argument('--year', type=int, default=2022,
                        help='Year for GeoTessera embeddings (default: 2022)')
    parser.add_argument('--agbd-max', type=float, default=500.0,
                        help='Maximum AGBD threshold (default: 500 Mg/ha)')
    parser.add_argument('--n-iterations', type=int, default=15,
                        help='Number of active learning iterations (default: 15)')
    parser.add_argument('--samples-per-iter', type=int, default=10,
                        help='Samples per iteration (default: 10)')
    parser.add_argument('--n-pool', type=int, default=None,
                        help='Pool size for active learning (default: None = half of remaining data)')

    # Execution
    parser.add_argument('--output-dir', type=str, required=True,
                        help='Output directory for sweep results')
    parser.add_argument('--cache-dir', type=str, default='./cache',
                        help='Cache directory (shared across runs)')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device to use (default: cuda)')
    parser.add_argument('--parallel', action='store_true',
                        help='Run experiments in parallel')
    parser.add_argument('--n-workers', type=int, default=4,
                        help='Number of parallel workers (default: 4)')
    parser.add_argument('--dry-run', action='store_true',
                        help='Print commands without running')

    args = parser.parse_args()

    # Setup paths
    output_dir = Path(args.output_dir)
    cache_dir = Path(args.cache_dir)

    output_dir.mkdir(parents=True, exist_ok=True)

    # Generate experiment configurations
    print("Generating experiment configurations...")
    configs = generate_experiment_configs(
        seed_sizes=args.seed_sizes,
        n_random_seeds=args.n_random_seeds,
        strategies=args.strategies,
        bbox=args.bbox,
        year=args.year,
        agbd_max=args.agbd_max,
        n_iterations=args.n_iterations,
        samples_per_iter=args.samples_per_iter,
        n_pool=args.n_pool,
        base_output_dir=output_dir
    )

    print(f"\nSweep Configuration:")
    print(f"  Seed sizes: {args.seed_sizes}")
    print(f"  Random seeds per config: {args.n_random_seeds}")
    print(f"  Strategies: {args.strategies}")
    print(f"  Total experiments: {len(configs)}")
    print(f"  Output directory: {output_dir}")

    # Save sweep configuration
    sweep_config = {
        'seed_sizes': args.seed_sizes,
        'n_random_seeds': args.n_random_seeds,
        'strategies': args.strategies,
        'bbox': args.bbox,
        'year': args.year,
        'agbd_max': args.agbd_max,
        'n_iterations': args.n_iterations,
        'samples_per_iter': args.samples_per_iter,
        'n_pool': args.n_pool,
        'n_experiments': len(configs),
        'timestamp': datetime.now().isoformat()
    }

    with open(output_dir / 'sweep_config.json', 'w') as f:
        json.dump(sweep_config, f, indent=2)

    if args.dry_run:
        print("\n[DRY RUN MODE - No experiments will be executed]")

    # Run experiments
    if args.parallel and not args.dry_run:
        results = run_sweep_parallel(
            configs, cache_dir, args.device, args.n_workers, args.dry_run
        )
    else:
        results = run_sweep_sequential(
            configs, cache_dir, args.device, args.dry_run
        )

    # Save results summary
    if not args.dry_run:
        save_sweep_results(results, output_dir)
        print(f"\n{'='*60}")
        print("Sweep Complete!")
        print(f"Results saved to: {output_dir}")
        print(f"{'='*60}")
    else:
        print(f"\n[DRY RUN] Generated {len(configs)} experiment configurations")
        print("Run without --dry-run to execute experiments")


if __name__ == '__main__':
    main()
