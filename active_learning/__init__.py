"""
Active learning for GEDI Neural Process models.

This module provides:
- Sampling strategies (random, uncertainty, spatial, hybrid)
- Active learning loop framework
- Utilities for pool-based active learning
"""

from .strategies import (
    RandomSampler,
    UncertaintySampler,
    SpatialSampler,
    HybridSampler,
    get_sampler,
)

from .loop import ActiveLearningLoop
from .simple_loop import SimpleActiveLearningLoop

__all__ = [
    'RandomSampler',
    'UncertaintySampler',
    'SpatialSampler',
    'HybridSampler',
    'get_sampler',
    'ActiveLearningLoop',
    'SimpleActiveLearningLoop',
]
