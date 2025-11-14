"""
Model initialization and checkpoint loading utilities.

This module provides centralized functions for initializing GEDI Neural Process
models and loading checkpoints across the codebase.
"""

from pathlib import Path
from typing import Dict, Any, Optional, Tuple
import torch
from models.neural_process import GEDINeuralProcess


def initialize_model(
    config: Dict[str, Any],
    device: str = 'cpu'
) -> GEDINeuralProcess:
    """
    Initialize a GEDI Neural Process model from configuration.

    Args:
        config: Configuration dictionary containing model parameters
        device: Device to place model on ('cuda' or 'cpu')

    Returns:
        Initialized GEDINeuralProcess model
    """
    model = GEDINeuralProcess(
        patch_size=config.get('patch_size', 3),
        embedding_channels=128,
        embedding_feature_dim=config.get('embedding_feature_dim', 128),
        context_repr_dim=config.get('context_repr_dim', 128),
        hidden_dim=config.get('hidden_dim', 512),
        latent_dim=config.get('latent_dim', 128),
        output_uncertainty=True,
        architecture_mode=config.get('architecture_mode', 'deterministic'),
        num_attention_heads=config.get('num_attention_heads', 4)
    ).to(device)

    return model


def load_checkpoint(
    checkpoint_dir: Path,
    device: str = 'cpu',
    checkpoint_name: Optional[str] = None
) -> Tuple[Dict[str, Any], Path]:
    """
    Load a model checkpoint with automatic fallback.

    Args:
        checkpoint_dir: Directory containing checkpoint files
        device: Device to load checkpoint to
        checkpoint_name: Specific checkpoint filename (optional).
                        If None, tries best_r2_model.pt then best_model.pt

    Returns:
        Tuple of (checkpoint_dict, checkpoint_path)

    Raises:
        FileNotFoundError: If no valid checkpoint is found
    """
    if checkpoint_name:
        # Use specific checkpoint if provided
        checkpoint_path = checkpoint_dir / checkpoint_name
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found at {checkpoint_path}")
    else:
        # Try fallback order: best_r2_model.pt -> best_model.pt
        checkpoint_files = ['best_r2_model.pt', 'best_model.pt']
        checkpoint_path = None

        for ckpt_file in checkpoint_files:
            path = checkpoint_dir / ckpt_file
            if path.exists():
                checkpoint_path = path
                break

        if checkpoint_path is None:
            raise FileNotFoundError(
                f"No checkpoint found in {checkpoint_dir}. "
                f"Looked for: {checkpoint_files}"
            )

    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

    return checkpoint, checkpoint_path


def load_model_from_checkpoint(
    checkpoint_dir: Path,
    device: str = 'cpu',
    checkpoint_name: Optional[str] = None
) -> Tuple[GEDINeuralProcess, Dict[str, Any], Path]:
    """
    Load a complete model from checkpoint directory.

    This is a convenience function that:
    1. Loads the config
    2. Initializes the model
    3. Loads the checkpoint weights

    Args:
        checkpoint_dir: Directory containing config.json and checkpoint
        device: Device to load model to
        checkpoint_name: Specific checkpoint filename (optional)

    Returns:
        Tuple of (model, checkpoint_dict, checkpoint_path)

    Raises:
        FileNotFoundError: If config or checkpoint not found
    """
    from .config import load_config

    # Load config
    config_path = checkpoint_dir / 'config.json'
    config = load_config(config_path)

    # Initialize model
    model = initialize_model(config, device)

    # Load checkpoint
    checkpoint, checkpoint_path = load_checkpoint(
        checkpoint_dir, device, checkpoint_name
    )

    # Load weights
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    return model, checkpoint, checkpoint_path
