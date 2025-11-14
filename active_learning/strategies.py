"""
Sampling strategies for active learning.

This module implements different strategies for selecting which samples to acquire
from a pool of unlabeled data:
- Random: baseline random sampling
- Uncertainty: select samples with highest predictive uncertainty
- Spatial: maximize geographic diversity
- Hybrid: combine uncertainty and spatial diversity
"""

import numpy as np
from typing import Tuple, Optional
from abc import ABC, abstractmethod
from scipy.spatial.distance import cdist
from sklearn.cluster import KMeans


class SamplingStrategy(ABC):
    """Base class for sampling strategies."""

    def __init__(self, name: str):
        self.name = name

    @abstractmethod
    def select_samples(
        self,
        pool_indices: np.ndarray,
        pool_coords: np.ndarray,
        n_samples: int,
        uncertainties: Optional[np.ndarray] = None,
        train_coords: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """
        Select samples from the pool.

        Args:
            pool_indices: Indices of pool samples in the original dataset
            pool_coords: Coordinates of pool samples (n_pool, 2) [lon, lat]
            n_samples: Number of samples to select
            uncertainties: Predicted uncertainties for pool samples (n_pool,)
            train_coords: Coordinates of current training samples (n_train, 2)

        Returns:
            Selected indices from pool_indices
        """
        pass


class RandomSampler(SamplingStrategy):
    """Random sampling baseline."""

    def __init__(self, seed: Optional[int] = None):
        super().__init__("random")
        self.seed = seed
        self.rng = np.random.RandomState(seed)

    def select_samples(
        self,
        pool_indices: np.ndarray,
        pool_coords: np.ndarray,
        n_samples: int,
        uncertainties: Optional[np.ndarray] = None,
        train_coords: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """Select random samples from pool."""
        n_samples = min(n_samples, len(pool_indices))
        selected_idx = self.rng.choice(len(pool_indices), size=n_samples, replace=False)
        return pool_indices[selected_idx]


class UncertaintySampler(SamplingStrategy):
    """Uncertainty-based sampling (epistemic uncertainty)."""

    def __init__(self):
        super().__init__("uncertainty")

    def select_samples(
        self,
        pool_indices: np.ndarray,
        pool_coords: np.ndarray,
        n_samples: int,
        uncertainties: Optional[np.ndarray] = None,
        train_coords: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """Select samples with highest uncertainty."""
        if uncertainties is None:
            raise ValueError("UncertaintySampler requires uncertainties")

        n_samples = min(n_samples, len(pool_indices))

        # Select top-k uncertain samples
        top_k_idx = np.argsort(uncertainties)[-n_samples:]
        return pool_indices[top_k_idx]


class SpatialSampler(SamplingStrategy):
    """Spatial diversity sampling - maximize geographic spread."""

    def __init__(self, method: str = 'maxmin'):
        """
        Initialize spatial sampler.

        Args:
            method: Sampling method
                - 'maxmin': iteratively select point farthest from training set
                - 'kmeans': cluster pool points and sample cluster centers
        """
        super().__init__("spatial")
        self.method = method

    def select_samples(
        self,
        pool_indices: np.ndarray,
        pool_coords: np.ndarray,
        n_samples: int,
        uncertainties: Optional[np.ndarray] = None,
        train_coords: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """Select spatially diverse samples."""
        n_samples = min(n_samples, len(pool_indices))

        if self.method == 'maxmin':
            return self._maxmin_sampling(
                pool_indices, pool_coords, n_samples, train_coords
            )
        elif self.method == 'kmeans':
            return self._kmeans_sampling(
                pool_indices, pool_coords, n_samples
            )
        else:
            raise ValueError(f"Unknown spatial sampling method: {self.method}")

    def _maxmin_sampling(
        self,
        pool_indices: np.ndarray,
        pool_coords: np.ndarray,
        n_samples: int,
        train_coords: Optional[np.ndarray]
    ) -> np.ndarray:
        """
        Greedy maxmin sampling: iteratively select point farthest from
        all previously selected points.
        """
        selected_mask = np.zeros(len(pool_indices), dtype=bool)

        # Initialize with existing training points if provided
        if train_coords is not None and len(train_coords) > 0:
            # Compute distances to training set
            distances = cdist(pool_coords, train_coords).min(axis=1)
        else:
            # Start with arbitrary point
            distances = np.ones(len(pool_indices)) * np.inf
            distances[0] = 0

        # Greedily select farthest points
        for _ in range(n_samples):
            # Select point with maximum distance to nearest selected point
            farthest_idx = distances.argmax()
            selected_mask[farthest_idx] = True

            # Update distances
            new_distances = np.linalg.norm(
                pool_coords - pool_coords[farthest_idx],
                axis=1
            )
            distances = np.minimum(distances, new_distances)
            distances[selected_mask] = -np.inf  # Mark as selected

        return pool_indices[selected_mask]

    def _kmeans_sampling(
        self,
        pool_indices: np.ndarray,
        pool_coords: np.ndarray,
        n_samples: int
    ) -> np.ndarray:
        """
        K-means clustering: cluster pool points and select nearest point
        to each cluster center.
        """
        # Run k-means
        kmeans = KMeans(n_clusters=n_samples, random_state=42)
        kmeans.fit(pool_coords)

        # Find nearest pool point to each cluster center
        selected_idx = []
        for center in kmeans.cluster_centers_:
            distances = np.linalg.norm(pool_coords - center, axis=1)
            nearest_idx = distances.argmin()
            selected_idx.append(nearest_idx)

        selected_idx = np.array(selected_idx)
        return pool_indices[selected_idx]


class HybridSampler(SamplingStrategy):
    """
    Hybrid sampling: combine uncertainty and spatial diversity.

    Strategy: From top-k uncertain points, select n_samples that maximize
    spatial diversity.
    """

    def __init__(
        self,
        uncertainty_percentile: float = 0.75,
        spatial_method: str = 'maxmin'
    ):
        """
        Initialize hybrid sampler.

        Args:
            uncertainty_percentile: Keep top p% uncertain points for spatial selection
            spatial_method: Method for spatial sampling ('maxmin' or 'kmeans')
        """
        super().__init__("hybrid")
        self.uncertainty_percentile = uncertainty_percentile
        self.spatial_sampler = SpatialSampler(method=spatial_method)

    def select_samples(
        self,
        pool_indices: np.ndarray,
        pool_coords: np.ndarray,
        n_samples: int,
        uncertainties: Optional[np.ndarray] = None,
        train_coords: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """Select samples combining uncertainty and spatial diversity."""
        if uncertainties is None:
            raise ValueError("HybridSampler requires uncertainties")

        n_samples = min(n_samples, len(pool_indices))

        # Filter to top uncertain points
        n_candidates = max(
            n_samples * 2,
            int(len(pool_indices) * (1 - self.uncertainty_percentile))
        )
        n_candidates = min(n_candidates, len(pool_indices))

        top_uncertain_idx = np.argsort(uncertainties)[-n_candidates:]

        # Apply spatial sampling to uncertain candidates
        candidate_indices = pool_indices[top_uncertain_idx]
        candidate_coords = pool_coords[top_uncertain_idx]

        selected = self.spatial_sampler.select_samples(
            candidate_indices,
            candidate_coords,
            n_samples,
            uncertainties=None,
            train_coords=train_coords
        )

        return selected


def get_sampler(strategy_name: str, **kwargs) -> SamplingStrategy:
    """
    Factory function to get a sampling strategy by name.

    Args:
        strategy_name: Name of strategy ('random', 'uncertainty', 'spatial', 'hybrid')
        **kwargs: Additional arguments for the strategy

    Returns:
        SamplingStrategy instance
    """
    if strategy_name == 'random':
        return RandomSampler(**kwargs)
    elif strategy_name == 'uncertainty':
        return UncertaintySampler()
    elif strategy_name == 'spatial':
        return SpatialSampler(**kwargs)
    elif strategy_name == 'hybrid':
        return HybridSampler(**kwargs)
    else:
        raise ValueError(f"Unknown sampling strategy: {strategy_name}")
