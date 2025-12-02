"""
Ensemble module for combining natural, high-variance, and low-variance models.

This module implements a redundancy-based ensemble that leverages the frequency-domain
separation of image components. The ensemble combines three models:
1. Natural model: Trained on original CIFAR-10 images
2. High-variance model: Trained on low-pass filtered (high-variance) images
3. Low-variance model: Trained on high-pass filtered (low-variance) images

The ensemble supports two voting modes:
- Weak: Majority vote (2 out of 3 agree)
- Strong: Unanimous vote (all 3 must agree)
"""

from ensemble.ensemble import RedundancyEnsemble, NoClearWinner

__all__ = ['RedundancyEnsemble', 'NoClearWinner']

