"""
Active learning loop framework.

This module provides the main ActiveLearningLoop class that orchestrates
the iterative process of:
1. Training a model on current data
2. Querying uncertainties on pool
3. Selecting samples using a strategy
4. Adding selected samples to training set
5. Evaluating and logging progress
"""

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from tqdm import tqdm

from .strategies import SamplingStrategy
from models.neural_process import GEDINeuralProcess, neural_process_loss
from data.dataset import GEDINeuralProcessDataset, collate_neural_process
from utils.evaluation import compute_metrics
from utils.normalization import normalize_coords


class ActiveLearningLoop:
    """
    Active learning loop for Neural Process models.

    This class manages the iterative process of active learning:
    - Maintains training and pool sets
    - Trains models at each iteration
    - Queries uncertainties and selects samples
    - Tracks metrics over iterations
    """

    def __init__(
        self,
        model: GEDINeuralProcess,
        optimizer: torch.optim.Optimizer,
        device: torch.device,
        sampler: SamplingStrategy,
        batch_size: int = 32,
        epochs_per_iteration: int = 50,
        kl_weight: float = 0.1,
        global_bounds: Optional[Tuple[float, float, float, float]] = None,
        verbose: bool = True
    ):
        """
        Initialize active learning loop.

        Args:
            model: Neural process model
            optimizer: Optimizer for training
            device: Device to run on
            sampler: Sampling strategy
            batch_size: Batch size for training
            epochs_per_iteration: Number of epochs to train at each AL iteration
            kl_weight: Weight for KL divergence in loss
            global_bounds: Global coordinate bounds (lon_min, lat_min, lon_max, lat_max)
            verbose: Whether to print progress
        """
        self.model = model
        self.optimizer = optimizer
        self.device = device
        self.sampler = sampler
        self.batch_size = batch_size
        self.epochs_per_iteration = epochs_per_iteration
        self.kl_weight = kl_weight
        self.global_bounds = global_bounds
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
            save_dir: Directory to save checkpoints (optional)

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
            print(f"Active Learning with {self.sampler.name} strategy")
            print(f"Initial train: {len(train_df)}, Pool: {len(pool_df)}, Test: {len(test_df)}")
            print(f"{'='*60}\n")

        for iteration in range(n_iterations):
            if self.verbose:
                print(f"\n--- Iteration {iteration + 1}/{n_iterations} ---")
                print(f"Training samples: {len(train_df)}")

            # Train model on current training set
            train_loss, train_rmse = self._train_model(train_df)

            # Evaluate on test set
            test_metrics = self._evaluate_model(test_df)

            # Log metrics
            self.history['iteration'].append(iteration)
            self.history['n_train'].append(len(train_df))
            self.history['train_loss'].append(train_loss)
            self.history['train_rmse'].append(train_rmse)
            self.history['test_rmse'].append(test_metrics['rmse'])
            self.history['test_mae'].append(test_metrics['mae'])
            self.history['test_r2'].append(test_metrics['r2'])

            if self.verbose:
                print(f"Train Loss: {train_loss:.4f}, Train RMSE: {train_rmse:.4f}")
                print(f"Test RMSE: {test_metrics['rmse']:.4f}, MAE: {test_metrics['mae']:.4f}, R²: {test_metrics['r2']:.4f}")

            # Save checkpoint
            if save_dir:
                checkpoint_path = save_dir / f"checkpoint_iter_{iteration}.pt"
                torch.save({
                    'iteration': iteration,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'history': self.history,
                }, checkpoint_path)

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
            print("Active Learning Complete")
            print(f"{'='*60}\n")

        return self.history

    def _train_model(self, train_df: pd.DataFrame) -> Tuple[float, float]:
        """
        Train model on current training set.

        Returns:
            (train_loss, train_rmse)
        """
        # Create dataset and dataloader
        dataset = GEDINeuralProcessDataset(
            train_df,
            global_bounds=self.global_bounds,
            min_shots_per_tile=5,
            max_shots_per_tile=None
        )

        dataloader = DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=True,
            collate_fn=collate_neural_process,
            num_workers=0
        )

        self.model.train()
        total_loss = 0.0
        all_predictions = []
        all_targets = []

        for epoch in range(self.epochs_per_iteration):
            epoch_loss = 0.0
            n_batches = 0

            for batch in dataloader:
                for i in range(len(batch['context_coords'])):
                    self.optimizer.zero_grad()

                    # Move to device
                    context_coords = batch['context_coords'][i].to(self.device)
                    context_embeddings = batch['context_embeddings'][i].to(self.device)
                    context_agbd = batch['context_agbd'][i].to(self.device)
                    target_coords = batch['target_coords'][i].to(self.device)
                    target_embeddings = batch['target_embeddings'][i].to(self.device)
                    target_agbd = batch['target_agbd'][i].to(self.device)

                    # Forward pass
                    pred_mean, pred_log_var, z_mu, z_log_sigma = self.model(
                        context_coords,
                        context_embeddings,
                        context_agbd,
                        target_coords,
                        target_embeddings,
                        training=True
                    )

                    # Compute loss
                    loss, _ = neural_process_loss(
                        pred_mean, pred_log_var, target_agbd,
                        z_mu, z_log_sigma, self.kl_weight
                    )

                    # Backward pass
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                    self.optimizer.step()

                    epoch_loss += loss.item()
                    n_batches += 1

                    # Store predictions for RMSE
                    all_predictions.extend(pred_mean.detach().cpu().numpy().flatten())
                    all_targets.extend(target_agbd.detach().cpu().numpy().flatten())

            if epoch == self.epochs_per_iteration - 1:
                total_loss = epoch_loss / max(n_batches, 1)

        # Compute training RMSE
        predictions = np.array(all_predictions)
        targets = np.array(all_targets)
        train_rmse = np.sqrt(np.mean((predictions - targets) ** 2))

        return total_loss, train_rmse

    def _evaluate_model(self, test_df: pd.DataFrame) -> Dict[str, float]:
        """
        Evaluate model on test set.

        Returns:
            Dictionary with metrics: rmse, mae, r2
        """
        # Create dataset
        dataset = GEDINeuralProcessDataset(
            test_df,
            global_bounds=self.global_bounds,
            min_shots_per_tile=5,
            max_shots_per_tile=None
        )

        dataloader = DataLoader(
            dataset,
            batch_size=1,
            shuffle=False,
            collate_fn=collate_neural_process,
            num_workers=0
        )

        self.model.eval()
        all_predictions = []
        all_targets = []

        with torch.no_grad():
            for batch in dataloader:
                for i in range(len(batch['context_coords'])):
                    context_coords = batch['context_coords'][i].to(self.device)
                    context_embeddings = batch['context_embeddings'][i].to(self.device)
                    context_agbd = batch['context_agbd'][i].to(self.device)
                    target_coords = batch['target_coords'][i].to(self.device)
                    target_embeddings = batch['target_embeddings'][i].to(self.device)
                    target_agbd = batch['target_agbd'][i].to(self.device)

                    # Forward pass
                    pred_mean, _, _, _ = self.model(
                        context_coords,
                        context_embeddings,
                        context_agbd,
                        target_coords,
                        target_embeddings,
                        training=False
                    )

                    all_predictions.extend(pred_mean.cpu().numpy().flatten())
                    all_targets.extend(target_agbd.cpu().numpy().flatten())

        # Compute metrics
        predictions = np.array(all_predictions)
        targets = np.array(all_targets)
        metrics = compute_metrics(predictions, targets)

        return metrics

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
        # Get uncertainties for pool if needed
        uncertainties = None
        if self.sampler.name in ['uncertainty', 'hybrid']:
            uncertainties = self._get_pool_uncertainties(train_df, pool_df)

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
        train_df: pd.DataFrame,
        pool_df: pd.DataFrame
    ) -> np.ndarray:
        """
        Get predictive uncertainties for all pool samples.

        This uses the full training set as context and queries each pool
        sample individually.

        Returns:
            Array of uncertainties (n_pool,)
        """
        self.model.eval()

        # Prepare context (all training data)
        context_coords = torch.tensor(
            train_df[['longitude', 'latitude']].values,
            dtype=torch.float32
        ).to(self.device)

        context_embeddings = torch.stack([
            torch.tensor(emb, dtype=torch.float32)
            for emb in train_df['embedding_patch'].values
        ]).to(self.device)

        context_agbd = torch.tensor(
            train_df['agbd'].values.reshape(-1, 1),
            dtype=torch.float32
        ).to(self.device)

        # Normalize coordinates if bounds provided
        if self.global_bounds:
            context_coords_norm = normalize_coords(
                context_coords.cpu().numpy(),
                self.global_bounds
            )
            context_coords = torch.tensor(context_coords_norm, dtype=torch.float32).to(self.device)

        # Query pool samples in batches
        uncertainties = []
        batch_size = 100

        with torch.no_grad():
            for i in tqdm(range(0, len(pool_df), batch_size), desc="Querying pool", disable=not self.verbose):
                batch_pool = pool_df.iloc[i:i+batch_size]

                query_coords = torch.tensor(
                    batch_pool[['longitude', 'latitude']].values,
                    dtype=torch.float32
                ).to(self.device)

                query_embeddings = torch.stack([
                    torch.tensor(emb, dtype=torch.float32)
                    for emb in batch_pool['embedding_patch'].values
                ]).to(self.device)

                # Normalize query coordinates
                if self.global_bounds:
                    query_coords_norm = normalize_coords(
                        query_coords.cpu().numpy(),
                        self.global_bounds
                    )
                    query_coords = torch.tensor(query_coords_norm, dtype=torch.float32).to(self.device)

                # Get predictions
                _, pred_log_var, _, _ = self.model(
                    context_coords,
                    context_embeddings,
                    context_agbd,
                    query_coords,
                    query_embeddings,
                    training=False
                )

                # Extract uncertainty (std)
                if pred_log_var is not None:
                    pred_std = torch.exp(0.5 * pred_log_var).cpu().numpy().flatten()
                else:
                    pred_std = np.zeros(len(batch_pool))

                uncertainties.extend(pred_std)

        return np.array(uncertainties)
