"""
Configuration loading and saving utilities.

This module provides centralized functions for handling configuration files
across the GEDI Neural Process codebase.
"""

import json
from pathlib import Path
from typing import Dict, Any, Optional


def load_config(config_path: Path) -> Dict[str, Any]:
    """
    Load configuration from a JSON file.

    Args:
        config_path: Path to the config.json file

    Returns:
        Dictionary containing configuration parameters

    Raises:
        FileNotFoundError: If config file doesn't exist
        json.JSONDecodeError: If config file is not valid JSON
    """
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found at {config_path}")

    with open(config_path, 'r') as f:
        config = json.load(f)

    return config


def save_config(config: Dict[str, Any], config_path: Path) -> None:
    """
    Save configuration to a JSON file.

    Args:
        config: Dictionary containing configuration parameters
        config_path: Path where config.json should be saved
    """
    config_path.parent.mkdir(parents=True, exist_ok=True)

    serializable_config = _make_serializable(config)

    with open(config_path, 'w') as f:
        json.dump(serializable_config, f, indent=2)


def _make_serializable(obj: Any) -> Any:
    """
    Recursively convert objects to JSON-serializable types.

    Args:
        obj: Object to convert

    Returns:
        JSON-serializable version of the object
    """
    import numpy as np

    if isinstance(obj, dict):
        return {key: _make_serializable(value) for key, value in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [_make_serializable(item) for item in obj]
    elif isinstance(obj, (np.integer, np.floating)):
        return obj.item()
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, Path):
        return str(obj)
    else:
        return obj


def get_global_bounds(config: Dict[str, Any]) -> Optional[tuple]:
    """
    Extract global coordinate bounds from config.

    Args:
        config: Configuration dictionary

    Returns:
        Tuple of (min_lon, min_lat, max_lon, max_lat) or None if not present
    """
    if 'global_bounds' in config:
        return tuple(config['global_bounds'])
    return None
