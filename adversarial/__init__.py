"""
Adversarial example generation module.

This module provides functionality to generate adversarial examples using
the robustness library, supporting both PGD and FGSM attacks.

Each model uses its CORRESPONDING dataset (natural or filtered), and
adversarial examples are generated for BOTH train and test sets.
"""

from adversarial.generate_adversarial import (
    generate_adversarial_dataset,
    generate_adversarial_for_split,
    run_adversarial_pipeline,
    load_dataset_for_model,
    get_dataset_paths,
    PGD_CONFIG,
    FGSM_CONFIG,
)

__all__ = [
    'generate_adversarial_dataset',
    'generate_adversarial_for_split',
    'run_adversarial_pipeline',
    'load_dataset_for_model',
    'get_dataset_paths',
    'PGD_CONFIG',
    'FGSM_CONFIG',
]

