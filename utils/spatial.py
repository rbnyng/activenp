"""
Spatial utilities for data splitting and blocking.

This module provides functions for creating spatially-blocked test sets,
which is the gold standard for evaluating geospatial models' ability to
extrapolate rather than just interpolate.
"""

import numpy as np
import pandas as pd
from typing import Tuple
from scipy.spatial.distance import cdist


def create_spatial_blocks(
    df: pd.DataFrame,
    n_blocks_lon: int = 4,
    n_blocks_lat: int = 4
) -> pd.DataFrame:
    """
    Divide spatial region into grid blocks and assign each sample to a block.

    Args:
        df: DataFrame with 'longitude' and 'latitude' columns
        n_blocks_lon: Number of blocks along longitude axis
        n_blocks_lat: Number of blocks along latitude axis

    Returns:
        DataFrame with added 'block_id' column
    """
    df = df.copy()

    # Get spatial bounds
    lon_min, lon_max = df['longitude'].min(), df['longitude'].max()
    lat_min, lat_max = df['latitude'].min(), df['latitude'].max()

    # Create grid edges
    lon_edges = np.linspace(lon_min, lon_max, n_blocks_lon + 1)
    lat_edges = np.linspace(lat_min, lat_max, n_blocks_lat + 1)

    # Assign each sample to a block
    lon_bins = np.digitize(df['longitude'], lon_edges) - 1
    lat_bins = np.digitize(df['latitude'], lat_edges) - 1

    # Clip to valid range (edge cases)
    lon_bins = np.clip(lon_bins, 0, n_blocks_lon - 1)
    lat_bins = np.clip(lat_bins, 0, n_blocks_lat - 1)

    # Create block ID (row-major ordering)
    df['block_id'] = lat_bins * n_blocks_lon + lon_bins

    return df


def select_contiguous_test_blocks(
    df: pd.DataFrame,
    test_fraction: float = 0.2,
    n_blocks_lon: int = 4,
    n_blocks_lat: int = 4,
    seed: int = 42
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Select a spatially contiguous region for test set.

    Strategy: Select a rectangular region of blocks (e.g., top-right quadrant)
    to create a spatially separated test set.

    Args:
        df: DataFrame with 'longitude' and 'latitude'
        test_fraction: Target fraction of data for test set
        n_blocks_lon: Number of blocks along longitude
        n_blocks_lat: Number of blocks along latitude
        seed: Random seed for block selection

    Returns:
        (train_indices, test_indices) - indices into original df
    """
    df = create_spatial_blocks(df, n_blocks_lon, n_blocks_lat)

    # Calculate target number of blocks for test set
    total_blocks = n_blocks_lon * n_blocks_lat
    n_test_blocks = max(1, int(total_blocks * test_fraction))

    # Try to create a rectangular contiguous region
    # Strategy: select corner region (easier to visualize and interpret)
    np.random.seed(seed)

    # Determine shape of test region (as rectangular as possible)
    aspect_ratio = n_blocks_lon / n_blocks_lat
    test_blocks_lon = max(1, int(np.sqrt(n_test_blocks * aspect_ratio)))
    test_blocks_lat = max(1, int(n_test_blocks / test_blocks_lon))

    # Ensure we get close to target
    while test_blocks_lon * test_blocks_lat < n_test_blocks and test_blocks_lon < n_blocks_lon:
        test_blocks_lon += 1
    while test_blocks_lon * test_blocks_lat < n_test_blocks and test_blocks_lat < n_blocks_lat:
        test_blocks_lat += 1

    # Randomly choose which corner: 0=top-left, 1=top-right, 2=bottom-left, 3=bottom-right
    corner = np.random.randint(0, 4)

    if corner == 0:  # Top-left
        lon_start, lat_start = 0, n_blocks_lat - test_blocks_lat
    elif corner == 1:  # Top-right
        lon_start, lat_start = n_blocks_lon - test_blocks_lon, n_blocks_lat - test_blocks_lat
    elif corner == 2:  # Bottom-left
        lon_start, lat_start = 0, 0
    else:  # Bottom-right
        lon_start, lat_start = n_blocks_lon - test_blocks_lon, 0

    # Generate test block IDs
    test_block_ids = []
    for lat_idx in range(lat_start, lat_start + test_blocks_lat):
        for lon_idx in range(lon_start, lon_start + test_blocks_lon):
            block_id = lat_idx * n_blocks_lon + lon_idx
            test_block_ids.append(block_id)

    # Split data
    test_mask = df['block_id'].isin(test_block_ids)
    test_indices = df.index[test_mask].values
    train_indices = df.index[~test_mask].values

    return train_indices, test_indices


def split_data_spatial_block(
    df: pd.DataFrame,
    n_seed: int = 100,
    n_pool: int = None,
    test_fraction: float = 0.2,
    n_blocks_lon: int = 4,
    n_blocks_lat: int = 4,
    seed: int = 42
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Split data with spatially-blocked test set (gold standard for geospatial ML).

    This ensures the test set is spatially separated from training data,
    evaluating the model's ability to extrapolate, not just interpolate.

    Args:
        df: Full dataset
        n_seed: Size of initial seed set
        n_pool: Size of pool for active learning (None = use remaining non-test)
        test_fraction: Fraction of data for spatially-blocked test set
        n_blocks_lon: Number of spatial blocks along longitude
        n_blocks_lat: Number of spatial blocks along latitude
        seed: Random seed

    Returns:
        (seed_df, pool_df, test_df)
    """
    np.random.seed(seed)

    # First: create spatially-blocked test set
    train_pool_indices, test_indices = select_contiguous_test_blocks(
        df,
        test_fraction=test_fraction,
        n_blocks_lon=n_blocks_lon,
        n_blocks_lat=n_blocks_lat,
        seed=seed
    )

    test_df = df.loc[test_indices].copy()

    # Remaining data for training and pool
    train_pool_df = df.loc[train_pool_indices].copy()

    # Shuffle and split into seed and pool
    train_pool_df = train_pool_df.sample(frac=1, random_state=seed).reset_index(drop=True)

    n_seed = min(n_seed, len(train_pool_df))
    seed_df = train_pool_df.iloc[:n_seed].copy()

    remaining = train_pool_df.iloc[n_seed:].copy()

    if n_pool is None:
        n_pool = len(remaining)  # Use all remaining as pool

    pool_df = remaining.iloc[:n_pool].copy()

    print(f"\nSpatial blocking info:")
    print(f"  Blocks: {n_blocks_lon} x {n_blocks_lat} = {n_blocks_lon * n_blocks_lat} total")
    print(f"  Test set: {len(test_df)} samples (spatially blocked)")
    print(f"  Train+Pool: {len(train_pool_df)} samples")
    print(f"  Seed: {len(seed_df)} samples")
    print(f"  Pool: {len(pool_df)} samples")

    # Calculate spatial statistics
    if len(test_df) > 0:
        test_coords = test_df[['longitude', 'latitude']].values
        train_coords = seed_df[['longitude', 'latitude']].values

        # Minimum distance from test set to initial training set
        distances = cdist(test_coords, train_coords)
        min_distance = distances.min()
        mean_min_distance = distances.min(axis=1).mean()

        print(f"  Spatial separation:")
        print(f"    Min distance (test to train): {min_distance:.4f}°")
        print(f"    Mean min distance: {mean_min_distance:.4f}°")

    return seed_df, pool_df, test_df
