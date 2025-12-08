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
    HybridProductSampler,
    get_sampler,
)

from .simple_loop import SimpleActiveLearningLoop
from .rf_loop import RFActiveLearningLoop

__all__ = [
    'RandomSampler',
    'UncertaintySampler',
    'SpatialSampler',
    'HybridSampler',
    'HybridProductSampler',
    'get_sampler',
    'SimpleActiveLearningLoop',
    'RFActiveLearningLoop',
]
