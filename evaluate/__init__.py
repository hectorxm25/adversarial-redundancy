"""
Evaluation pipeline for trained models.

This module provides utilities for:
- Evaluating trained ResNet18 models on test datasets
- Evaluating models on both natural and filtered test sets
- Comparing accuracies across different models
- Generating evaluation reports
"""

from .evaluate_models import evaluate_model, evaluate_all_models, run_evaluation_pipeline

__all__ = [
    'evaluate_model',
    'evaluate_all_models',
    'run_evaluation_pipeline',
]

