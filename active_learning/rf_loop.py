"""
Active learning loop for Random Forest baseline.

This module provides an active learning loop for Random Forest that uses
ensemble variance as uncertainty. This baseline demonstrates the failure
mode of uncalibrated uncertainty for active learning.
"""

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional
from tqdm import tqdm

from .strategies import SamplingStrategy
from models.random_forest_baseline import RandomForestBaseline


class RFActiveLearningLoop:
    """
    Active learning loop for Random Forest baseline.

    Uses ensemble variance as "uncertainty" to demonstrate the failure mode
    of poorly calibrated uncertainty for active learning.
    """

    def __init__(
        self,
        model: RandomForestBaseline,
        sampler: SamplingStrategy,
        verbose: bool = True
    ):
        """
        Initialize RF active learning loop.

        Args:
            model: Random Forest baseline model
            sampler: Sampling strategy
            verbose: Whether to print progress
        """
        self.model = model
        self.sampler = sampler
        self.verbose = verbose

        # History tracking
        self.history = {
            'iteration': [],
            'n_train': [],
            'train_loss': [],
            'train_rmse': [],
            'test_rmse': [],
            'test_mae': [],
            'test_r2': [],
            'selected_indices': []
        }

    def run(
        self,
        train_df: pd.DataFrame,
        pool_df: pd.DataFrame,
        test_df: pd.DataFrame,
        n_iterations: int,
        samples_per_iteration: int,
        save_dir: Optional[Path] = None
    ) -> Dict[str, List]:
        """
        Run active learning loop.

        Args:
            train_df: Initial training data (seed set)
            pool_df: Pool of unlabeled samples to select from
            test_df: Test set for evaluation
            n_iterations: Number of active learning iterations
            samples_per_iteration: Number of samples to acquire each iteration
            save_dir: Directory to save results (optional)

        Returns:
            Dictionary containing history of metrics
        """
        if save_dir:
            save_dir.mkdir(parents=True, exist_ok=True)

        # Make copies to avoid modifying originals
        train_df = train_df.copy()
        pool_df = pool_df.copy()

        if self.verbose:
            print(f"\n{'='*60}")
            print(f"RF Active Learning with {self.sampler.name} strategy")
            print(f"Initial train: {len(train_df)}, Pool: {len(pool_df)}, Test: {len(test_df)}")
            print(f"{'='*60}\n")

        for iteration in range(n_iterations):
            if self.verbose:
                print(f"\n--- Iteration {iteration + 1}/{n_iterations} ---")
                print(f"Training samples: {len(train_df)}")

            # Train model on current training set
            train_metrics = self.model.fit(train_df)

            # Evaluate on test set
            test_metrics = self.model.evaluate(test_df)

            # Log metrics
            self.history['iteration'].append(iteration)
            self.history['n_train'].append(len(train_df))
            self.history['train_loss'].append(train_metrics['train_loss'])
            self.history['train_rmse'].append(train_metrics['train_rmse'])
            self.history['test_rmse'].append(test_metrics['test_rmse'])
            self.history['test_mae'].append(test_metrics['test_mae'])
            self.history['test_r2'].append(test_metrics['test_r2'])

            if self.verbose:
                print(f"Train Loss: {train_metrics['train_loss']:.4f}, Train RMSE: {train_metrics['train_rmse']:.4f}")
                print(f"Test RMSE: {test_metrics['test_rmse']:.4f}, MAE: {test_metrics['test_mae']:.4f}, R²: {test_metrics['test_r2']:.4f}")

            # Stop if no more samples in pool
            if len(pool_df) == 0:
                if self.verbose:
                    print("\nPool exhausted. Stopping.")
                break

            # Select new samples from pool
            selected_indices = self._select_samples(
                train_df, pool_df, samples_per_iteration
            )

            self.history['selected_indices'].append(selected_indices.tolist())

            # Move selected samples from pool to train
            selected_samples = pool_df.loc[selected_indices]
            train_df = pd.concat([train_df, selected_samples], ignore_index=True)
            pool_df = pool_df.drop(selected_indices)

            if self.verbose:
                print(f"Selected {len(selected_indices)} samples. New train size: {len(train_df)}")

        if self.verbose:
            print(f"\n{'='*60}")
            print("RF Active Learning Complete")
            print(f"{'='*60}\n")

        return self.history

    def _select_samples(
        self,
        train_df: pd.DataFrame,
        pool_df: pd.DataFrame,
        n_samples: int
    ) -> np.ndarray:
        """
        Select samples from pool using the sampling strategy.

        Returns:
            Indices of selected samples (in pool_df.index)
        """
        # Get uncertainties for pool if needed (ensemble variance)
        uncertainties = None
        if self.sampler.name in ['uncertainty', 'hybrid', 'hybrid_product']:
            uncertainties = self._get_pool_uncertainties(pool_df)

        # Get coordinates
        pool_coords = pool_df[['longitude', 'latitude']].values
        train_coords = train_df[['longitude', 'latitude']].values if len(train_df) > 0 else None

        # Select samples
        pool_indices = pool_df.index.values
        selected_indices = self.sampler.select_samples(
            pool_indices=pool_indices,
            pool_coords=pool_coords,
            n_samples=n_samples,
            uncertainties=uncertainties,
            train_coords=train_coords
        )

        return selected_indices

    def _get_pool_uncertainties(
        self,
        pool_df: pd.DataFrame
    ) -> np.ndarray:
        """
        Get ensemble variance uncertainties for all pool samples.

        This is the "bad" uncertainty - it captures aleatoric noise
        rather than epistemic uncertainty.

        Returns:
            Array of uncertainties (n_pool,) - ensemble standard deviation
        """
        if self.verbose:
            print("  Computing RF ensemble uncertainties...")

        # Get predictions and uncertainties
        _, uncertainties = self.model.predict(pool_df, return_uncertainty=True)

        return uncertainties
