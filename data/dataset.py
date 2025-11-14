"""
PyTorch Dataset for Neural Process training with GEDI + GeoTessera embeddings.
"""

import torch
from torch.utils.data import Dataset
import numpy as np
import pandas as pd
from typing import Optional, Dict, Tuple
import random

from utils.normalization import normalize_coords


class GEDINeuralProcessDataset(Dataset):
    """
    Dataset for Neural Process training with GEDI shots and embeddings.

    Each sample is a tile with multiple GEDI shots. The dataset creates
    context/target splits for Neural Process training.
    """

    def __init__(
        self,
        data_df: pd.DataFrame,
        min_shots_per_tile: int = 10,
        max_shots_per_tile: Optional[int] = None,
        context_ratio_range: Tuple[float, float] = (0.3, 0.7),
        normalize_coords: bool = True,
        normalize_agbd: bool = True,
        agbd_scale: float = 200.0,  # Typical max AGBD in Mg/ha
        log_transform_agbd: bool = True,
        augment_coords: bool = True,
        coord_noise_std: float = 0.01,
        global_bounds: Optional[Tuple[float, float, float, float]] = None
    ):
        """
        Initialize dataset.

        Args:
            data_df: DataFrame with columns: latitude, longitude, agbd, embedding_patch, tile_id
            min_shots_per_tile: Minimum number of shots per tile to include
            max_shots_per_tile: Maximum shots per tile (subsample if exceeded)
            context_ratio_range: Range of context/total ratios for training (min, max)
            normalize_coords: Normalize coordinates to [0, 1] using global bounds
            normalize_agbd: Normalize AGBD values
            agbd_scale: Scale factor for AGBD normalization
            log_transform_agbd: Apply log(1+x) transform to AGBD
            augment_coords: Add small random noise to coordinates during training
            coord_noise_std: Standard deviation of coordinate noise
            global_bounds: Global coordinate bounds (lon_min, lat_min, lon_max, lat_max).
                          If None, computed from data_df. Should be computed from training
                          data and shared across train/val/test for proper normalization.
        """
        self.data_df = data_df[data_df['embedding_patch'].notna()].copy()

        self.tiles = []
        for tile_id, group in self.data_df.groupby('tile_id'):
            if len(group) >= min_shots_per_tile:
                # Subsample if too many shots
                if max_shots_per_tile and len(group) > max_shots_per_tile:
                    group = group.sample(n=max_shots_per_tile, random_state=42)
                self.tiles.append(group)

        self.min_shots_per_tile = min_shots_per_tile
        self.max_shots_per_tile = max_shots_per_tile
        self.context_ratio_range = context_ratio_range
        self.normalize_coords = normalize_coords
        self.normalize_agbd = normalize_agbd
        self.agbd_scale = agbd_scale
        self.log_transform_agbd = log_transform_agbd
        self.augment_coords = augment_coords
        self.coord_noise_std = coord_noise_std

        if global_bounds is None:
            self.lon_min = self.data_df['longitude'].min()
            self.lon_max = self.data_df['longitude'].max()
            self.lat_min = self.data_df['latitude'].min()
            self.lat_max = self.data_df['latitude'].max()
        else:
            self.lon_min, self.lat_min, self.lon_max, self.lat_max = global_bounds

        print(f"Dataset initialized with {len(self.tiles)} tiles")
        if len(self.tiles) > 0:
            shots_per_tile = [len(t) for t in self.tiles]
            print(f"Shots per tile: min={min(shots_per_tile)}, "
                  f"max={max(shots_per_tile)}, mean={np.mean(shots_per_tile):.1f}")

    def __len__(self) -> int:
        return len(self.tiles)

    def _normalize_coordinates(self, coords: np.ndarray) -> np.ndarray:
        """
        Normalize coordinates to [0, 1] range using global bounds.

        This ensures the model can learn latitude-dependent patterns (e.g., climate zones)
        since coordinates are normalized consistently across all tiles.

        Args:
            coords: (N, 2) array of [lon, lat]

        Returns:
            Normalized coordinates (N, 2)
        """
        # Use global bounds for normalization
        global_bounds = (self.lon_min, self.lat_min, self.lon_max, self.lat_max)
        return normalize_coords(coords, global_bounds)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Get a training sample.

        Returns a dict with:
            - context_coords: (n_context, 2) coordinates [lon, lat]
            - context_embeddings: (n_context, patch_size, patch_size, 128)
            - context_agbd: (n_context, 1) AGBD values
            - target_coords: (n_target, 2) coordinates
            - target_embeddings: (n_target, patch_size, patch_size, 128)
            - target_agbd: (n_target, 1) AGBD values
        """
        tile_data = self.tiles[idx].copy()
        n_shots = len(tile_data)

        context_ratio = random.uniform(*self.context_ratio_range)
        n_context = max(1, int(n_shots * context_ratio))

        context_indices = random.sample(range(n_shots), n_context)
        target_indices = [i for i in range(n_shots) if i not in context_indices]

        tile_array = tile_data.to_numpy()
        coords = tile_data[['longitude', 'latitude']].values
        embeddings = np.stack(tile_data['embedding_patch'].values)
        agbd = tile_data['agbd'].values[:, None]

        if self.normalize_coords:
            coords = self._normalize_coordinates(coords)

        if self.augment_coords:
            coords = coords + np.random.normal(0, self.coord_noise_std, coords.shape)
            coords = np.clip(coords, 0, 1)

        if self.normalize_agbd:
            if self.log_transform_agbd:
                agbd = np.log1p(agbd) / np.log1p(self.agbd_scale)
            else:
                agbd = agbd / self.agbd_scale

        context_coords = coords[context_indices]
        context_embeddings = embeddings[context_indices]
        context_agbd = agbd[context_indices]

        target_coords = coords[target_indices]
        target_embeddings = embeddings[target_indices]
        target_agbd = agbd[target_indices]

        return {
            'context_coords': torch.from_numpy(context_coords).float(),
            'context_embeddings': torch.from_numpy(context_embeddings).float(),
            'context_agbd': torch.from_numpy(context_agbd).float(),
            'target_coords': torch.from_numpy(target_coords).float(),
            'target_embeddings': torch.from_numpy(target_embeddings).float(),
            'target_agbd': torch.from_numpy(target_agbd).float(),
        }


def collate_neural_process(batch):
    """
    Custom collate function for Neural Process batches.

    Handles variable numbers of context and target points across tiles.

    Args:
        batch: List of dicts from GEDINeuralProcessDataset

    Returns:
        Batched dict with lists of tensors (one per tile in batch)
    """
    # Since tiles can have different numbers of shots, we return lists
    return {
        'context_coords': [item['context_coords'] for item in batch],
        'context_embeddings': [item['context_embeddings'] for item in batch],
        'context_agbd': [item['context_agbd'] for item in batch],
        'target_coords': [item['target_coords'] for item in batch],
        'target_embeddings': [item['target_embeddings'] for item in batch],
        'target_agbd': [item['target_agbd'] for item in batch],
    }


class GEDIInferenceDataset(Dataset):
    """
    Dataset for inference on a dense grid.

    Used for predicting AGBD at arbitrary locations within a tile.
    """

    def __init__(
        self,
        context_df: pd.DataFrame,
        query_lons: np.ndarray,
        query_lats: np.ndarray,
        query_embeddings: np.ndarray,
        normalize_coords: bool = True,
        normalize_agbd: bool = True,
        agbd_scale: float = 200.0,
        log_transform_agbd: bool = True,
        global_bounds: Optional[Tuple[float, float, float, float]] = None
    ):
        """
        Initialize inference dataset.

        Args:
            context_df: DataFrame with context GEDI shots
            query_lons: Array of query longitudes
            query_lats: Array of query latitudes
            query_embeddings: Array of query embeddings (N, patch_size, patch_size, 128)
            normalize_coords: Normalize coordinates
            normalize_agbd: Normalize AGBD
            agbd_scale: AGBD scale factor
            log_transform_agbd: Apply log transform to AGBD
            global_bounds: Global coordinate bounds (lon_min, lat_min, lon_max, lat_max).
                          If None, computed from context + query data.
        """
        self.context_df = context_df[context_df['embedding_patch'].notna()].copy()
        self.query_lons = query_lons
        self.query_lats = query_lats
        self.query_embeddings = query_embeddings

        self.normalize_coords = normalize_coords
        self.normalize_agbd = normalize_agbd
        self.agbd_scale = agbd_scale
        self.log_transform_agbd = log_transform_agbd

        # Store global bounds for normalization
        if global_bounds is None:
            # Compute from context + query data
            all_lons = np.concatenate([context_df['longitude'].values, query_lons])
            all_lats = np.concatenate([context_df['latitude'].values, query_lats])
            self.lon_min = all_lons.min()
            self.lon_max = all_lons.max()
            self.lat_min = all_lats.min()
            self.lat_max = all_lats.max()
        else:
            # Use provided global bounds
            self.lon_min, self.lat_min, self.lon_max, self.lat_max = global_bounds

        # Prepare context data
        self.context_coords = self.context_df[['longitude', 'latitude']].values
        self.context_embeddings = np.stack(self.context_df['embedding_patch'].values)
        self.context_agbd = self.context_df['agbd'].values[:, None]

        # Normalize context
        if self.normalize_agbd:
            if self.log_transform_agbd:
                self.context_agbd = np.log1p(self.context_agbd) / np.log1p(self.agbd_scale)
            else:
                self.context_agbd = self.context_agbd / self.agbd_scale

    def __len__(self) -> int:
        return len(self.query_lons)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """Get query point with context."""
        query_coord = np.array([[self.query_lons[idx], self.query_lats[idx]]])
        query_embedding = self.query_embeddings[idx:idx+1]

        # Normalize if needed
        if self.normalize_coords:
            # Use global bounds for normalization
            lon_range = self.lon_max - self.lon_min if self.lon_max > self.lon_min else 1.0
            lat_range = self.lat_max - self.lat_min if self.lat_max > self.lat_min else 1.0

            context_coords_norm = self.context_coords.copy()
            context_coords_norm[:, 0] = (context_coords_norm[:, 0] - self.lon_min) / lon_range
            context_coords_norm[:, 1] = (context_coords_norm[:, 1] - self.lat_min) / lat_range

            query_coord_norm = query_coord.copy()
            query_coord_norm[:, 0] = (query_coord_norm[:, 0] - self.lon_min) / lon_range
            query_coord_norm[:, 1] = (query_coord_norm[:, 1] - self.lat_min) / lat_range
        else:
            context_coords_norm = self.context_coords
            query_coord_norm = query_coord

        return {
            'context_coords': torch.from_numpy(context_coords_norm).float(),
            'context_embeddings': torch.from_numpy(self.context_embeddings).float(),
            'context_agbd': torch.from_numpy(self.context_agbd).float(),
            'query_coord': torch.from_numpy(query_coord_norm).float(),
            'query_embedding': torch.from_numpy(query_embedding).float(),
        }
