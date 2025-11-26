"""
Training pipeline for adversarial robustness experiments.

This module provides utilities for:
- Training ResNet18 models using the robustness library
- Training on natural CIFAR-10 dataset
- Training on filtered (high-variance and low-variance) datasets
- GPU memory management for sequential training runs
"""

from .datasets import NumpyDataset, FilteredCIFAR10
from .train_models import train_single_model, run_training_pipeline

__all__ = [
    'NumpyDataset',
    'FilteredCIFAR10', 
    'train_single_model',
    'run_training_pipeline',
]

