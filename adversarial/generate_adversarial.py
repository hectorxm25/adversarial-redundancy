#!/usr/bin/env python3
"""
Adversarial example generation using the robustness library.

This script generates adversarial examples using PGD and FGSM attacks for
trained models. FGSM is implemented as a single-step PGD attack.

The robustness library's attack mechanism is used, which provides:
- PGD (Projected Gradient Descent) attack: Multi-step iterative attack
- FGSM (Fast Gradient Sign Method): Single-step attack (PGD with 1 iteration)

For each model, adversarial examples are generated using the CORRESPONDING
dataset that the model was trained on:
- Natural model: uses data/cifar10_natural/
- Filtered models: uses data/filtered_r{radius}/ with the corresponding variance type

Adversarial examples are generated for BOTH train and test sets.

Usage:
    # For a single model with explicit dataset paths
    python generate_adversarial.py --model-path ./models/resnet18_natural/checkpoint.pt.best \
                                   --train-data-dir ./data/cifar10_natural \
                                   --test-data-dir ./data/cifar10_natural \
                                   --dataset-type natural \
                                   --output-dir ./adversarial_data/resnet18_natural \
                                   --device cuda:0 \
                                   --attack-type pgd

    # For filtered model
    python generate_adversarial.py --model-path ./models/resnet18_high_variance_r15/checkpoint.pt.best \
                                   --train-data-dir ./data/filtered_r15/cifar10_train_high_variance_r15.0 \
                                   --test-data-dir ./data/filtered_r15/cifar10_test_high_variance_r15.0 \
                                   --dataset-type filtered \
                                   --output-dir ./adversarial_data/resnet18_high_variance_r15 \
                                   --device cuda:0 \
                                   --attack-type pgd
"""

import os
import sys
import gc
import argparse
import logging
import json
from pathlib import Path
from datetime import datetime
from typing import Tuple, Optional, Dict, Any, List

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from robustness import model_utils, attacker
from robustness.attacker import AttackerModel

from train.datasets import FilteredCIFAR10, load_filtered_dataset, load_natural_dataset, NumpyDataset
from torchvision import transforms

# =============================================================================
# GLOBAL ATTACK CONFIGURATIONS
# =============================================================================
# These can be modified to change attack parameters for all experiments

# PGD Attack Configuration
# - eps: Maximum perturbation size (in [0,1] range, where 8/255 ≈ 0.031)
# - step_size: Step size for each iteration (typically eps/4)
# - iterations: Number of attack iterations
# - constraint: Lp norm constraint ('inf' for L-infinity, '2' for L2)
# - random_start: Whether to start from a random point within the epsilon ball
PGD_CONFIG = {
    'eps': 8.0 / 255.0,          # 8/255 is a standard choice for CIFAR-10
    'step_size': 2.0 / 255.0,    # 2/255 step size
    'iterations': 20,             # 20 iterations for strong attack
    'constraint': 'inf',          # L-infinity norm
    'random_start': True,         # Random start within epsilon ball
}

# FGSM Attack Configuration
# FGSM is implemented as a single-step PGD attack
# - eps: Maximum perturbation size
# - constraint: Lp norm constraint
FGSM_CONFIG = {
    'eps': 8.0 / 255.0,          # Same as PGD for fair comparison
    'step_size': 8.0 / 255.0,    # Step size = eps for single-step attack
    'iterations': 1,              # Single step for FGSM
    'constraint': 'inf',          # L-infinity norm
    'random_start': False,        # No random start for pure FGSM
}

# Batch size for adversarial generation (can be adjusted based on GPU memory)
GENERATION_BATCH_SIZE = 64


def setup_logging(log_dir: str, run_name: str) -> logging.Logger:
    """
    Set up logging for adversarial generation.
    
    Args:
        log_dir: Directory for log files
        run_name: Name of the generation run
        
    Returns:
        Configured logger
    """
    log_path = Path(log_dir)
    log_path.mkdir(parents=True, exist_ok=True)
    
    # Create logger
    logger = logging.getLogger(run_name)
    logger.setLevel(logging.DEBUG)
    
    # Clear any existing handlers
    logger.handlers = []
    
    # File handler
    log_file = log_path / f"{run_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.DEBUG)
    
    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    
    # Formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)
    
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    return logger


def cleanup_gpu_memory():
    """
    Clean up GPU memory after processing.
    """
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()


def get_attack_config(attack_type: str) -> Dict[str, Any]:
    """
    Get attack configuration based on attack type.
    
    Args:
        attack_type: Type of attack ('pgd' or 'fgsm')
        
    Returns:
        Attack configuration dictionary
    """
    if attack_type.lower() == 'pgd':
        return PGD_CONFIG.copy()
    elif attack_type.lower() == 'fgsm':
        return FGSM_CONFIG.copy()
    else:
        raise ValueError(f"Unknown attack type: {attack_type}. Supported: 'pgd', 'fgsm'")


def generate_adversarial_for_split(
    model,
    images: np.ndarray,
    labels: np.ndarray,
    output_dir: str,
    split_name: str,
    attack_type: str,
    attack_config: Dict[str, Any],
    torch_device: torch.device,
    batch_size: int = GENERATION_BATCH_SIZE,
    workers: int = 4,
    logger: Optional[logging.Logger] = None,
) -> Dict[str, Any]:
    """
    Generate adversarial examples for a single data split (train or test).
    
    Args:
        model: Loaded model to attack
        images: Images to generate adversarial examples from (N, H, W, C)
        labels: Labels for the images (N,)
        output_dir: Directory to save adversarial dataset
        split_name: Name of the split ('train' or 'test')
        attack_type: Type of attack ('pgd' or 'fgsm')
        attack_config: Attack configuration dictionary
        torch_device: Device to use
        batch_size: Batch size for generation
        workers: Number of data loading workers
        logger: Optional logger for output
        
    Returns:
        Dictionary containing generation results and statistics for this split
    """
    if logger is None:
        logger = logging.getLogger('adversarial_generation')
    
    logger.info(f"  Generating {attack_type.upper()} adversarial examples for {split_name} set...")
    logger.info(f"  Number of images: {len(images)}")
    
    # Create output directory for this split
    output_path = Path(output_dir) / split_name
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Create dataloader
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(
            FilteredCIFAR10.CIFAR10_MEAN,
            FilteredCIFAR10.CIFAR10_STD
        ),
    ])
    
    dataset = NumpyDataset(images, labels, transform=test_transform)
    data_loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=workers,
        pin_memory=True,
    )
    
    logger.info(f"  Created dataloader with {len(data_loader)} batches")
    
    # Generate adversarial examples
    start_time = datetime.now()
    
    all_adv_images = []
    all_labels = []
    all_clean_images = []
    
    # Track attack success
    total_correct_clean = 0
    total_correct_adv = 0
    total_samples = 0
    
    # Attack parameters for robustness library
    attack_kwargs = {
        'constraint': attack_config['constraint'],
        'eps': attack_config['eps'],
        'step_size': attack_config['step_size'],
        'iterations': attack_config['iterations'],
        'random_start': attack_config.get('random_start', False),
        'random_restarts': False,
        'do_tqdm': False,
    }
    
    with torch.no_grad():
        model.eval()
    
    for batch_idx, (batch_images, batch_labels) in enumerate(tqdm(data_loader, desc=f"  {split_name.capitalize()} {attack_type.upper()}")):
        batch_images = batch_images.to(torch_device)
        batch_labels = batch_labels.to(torch_device)
        
        # Get clean predictions
        with torch.no_grad():
            clean_output, _ = model(batch_images)
            clean_pred = clean_output.argmax(dim=1)
            total_correct_clean += (clean_pred == batch_labels).sum().item()
        
        # Generate adversarial examples
        with torch.enable_grad():
            _, adv_images = model(
                batch_images,
                batch_labels,
                make_adv=True,
                **attack_kwargs
            )
        
        # Get adversarial predictions
        with torch.no_grad():
            adv_output, _ = model(adv_images)
            adv_pred = adv_output.argmax(dim=1)
            total_correct_adv += (adv_pred == batch_labels).sum().item()
        
        total_samples += len(batch_labels)
        
        # Denormalize images to [0, 1] range for saving
        mean = torch.tensor(FilteredCIFAR10.CIFAR10_MEAN).view(1, 3, 1, 1).to(torch_device)
        std = torch.tensor(FilteredCIFAR10.CIFAR10_STD).view(1, 3, 1, 1).to(torch_device)
        
        adv_images_denorm = adv_images * std + mean
        clean_images_denorm = batch_images * std + mean
        
        # Clamp to [0, 1] range
        adv_images_denorm = torch.clamp(adv_images_denorm, 0, 1)
        clean_images_denorm = torch.clamp(clean_images_denorm, 0, 1)
        
        # Convert to numpy and store
        # Shape: (N, C, H, W) -> (N, H, W, C)
        adv_np = adv_images_denorm.cpu().numpy().transpose(0, 2, 3, 1)
        clean_np = clean_images_denorm.cpu().numpy().transpose(0, 2, 3, 1)
        labels_np = batch_labels.cpu().numpy()
        
        all_adv_images.append(adv_np)
        all_clean_images.append(clean_np)
        all_labels.append(labels_np)
    
    end_time = datetime.now()
    generation_time = (end_time - start_time).total_seconds()
    
    # Concatenate all results
    all_adv_images = np.concatenate(all_adv_images, axis=0)
    all_clean_images = np.concatenate(all_clean_images, axis=0)
    all_labels = np.concatenate(all_labels, axis=0)
    
    # Convert to uint8 for storage efficiency
    all_adv_images_uint8 = (all_adv_images * 255).astype(np.uint8)
    all_clean_images_uint8 = (all_clean_images * 255).astype(np.uint8)
    
    # Calculate statistics
    clean_accuracy = 100.0 * total_correct_clean / total_samples
    adv_accuracy = 100.0 * total_correct_adv / total_samples
    attack_success_rate = 100.0 - adv_accuracy
    
    logger.info(f"  {split_name.capitalize()} generation completed in {generation_time:.2f} seconds")
    logger.info(f"  {split_name.capitalize()} clean accuracy: {clean_accuracy:.2f}%")
    logger.info(f"  {split_name.capitalize()} adversarial accuracy: {adv_accuracy:.2f}%")
    logger.info(f"  {split_name.capitalize()} attack success rate: {attack_success_rate:.2f}%")
    
    # Save adversarial dataset
    logger.info(f"  Saving {split_name} adversarial dataset...")
    np.save(output_path / "adversarial_images.npy", all_adv_images_uint8)
    np.save(output_path / "labels.npy", all_labels)
    np.save(output_path / "clean_images.npy", all_clean_images_uint8)
    
    # Save metadata for this split
    split_metadata = {
        'split': split_name,
        'num_samples': len(all_labels),
        'clean_accuracy': clean_accuracy,
        'adversarial_accuracy': adv_accuracy,
        'attack_success_rate': attack_success_rate,
        'generation_time_seconds': generation_time,
    }
    
    with open(output_path / "metadata.json", 'w') as f:
        json.dump(split_metadata, f, indent=2)
    
    logger.info(f"  Saved {len(all_labels)} {split_name} adversarial examples to {output_path}")
    
    return split_metadata


def generate_adversarial_dataset(
    model_path: str,
    train_images: np.ndarray,
    train_labels: np.ndarray,
    test_images: np.ndarray,
    test_labels: np.ndarray,
    output_dir: str,
    attack_type: str,
    device: str,
    batch_size: int = GENERATION_BATCH_SIZE,
    workers: int = 4,
    logger: Optional[logging.Logger] = None,
    attack_config_override: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Generate adversarial examples for both train and test sets using a trained model.
    
    Args:
        model_path: Path to the model checkpoint
        train_images: Training images (N, H, W, C)
        train_labels: Training labels (N,)
        test_images: Test images (N, H, W, C)
        test_labels: Test labels (N,)
        output_dir: Directory to save adversarial dataset
        attack_type: Type of attack ('pgd' or 'fgsm')
        device: Device to use (e.g., 'cuda:0')
        batch_size: Batch size for generation
        workers: Number of data loading workers
        logger: Optional logger for output
        attack_config_override: Optional dict to override attack config parameters
        
    Returns:
        Dictionary containing generation results and statistics
    """
    if logger is None:
        logger = logging.getLogger('adversarial_generation')
    
    # Get attack configuration
    attack_config = get_attack_config(attack_type)
    if attack_config_override:
        attack_config.update(attack_config_override)
    
    logger.info("=" * 60)
    logger.info(f"Generating {attack_type.upper()} Adversarial Examples")
    logger.info("=" * 60)
    logger.info(f"Model: {model_path}")
    logger.info(f"Output: {output_dir}")
    logger.info(f"Device: {device}")
    logger.info(f"Attack config: {attack_config}")
    logger.info(f"Train images: {len(train_images)}, Test images: {len(test_images)}")
    
    # Parse device
    if device == 'cpu':
        torch_device = torch.device('cpu')
    else:
        # Extract GPU index from device string
        if device.isdigit():
            gpu_idx = int(device)
        elif device.startswith('cuda:'):
            gpu_idx = int(device.split(':')[1])
        else:
            gpu_idx = 0
        
        # Set CUDA device
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is not available but a CUDA device was requested")
        if gpu_idx >= torch.cuda.device_count():
            raise RuntimeError(f"GPU {gpu_idx} is not available. Available GPUs: 0-{torch.cuda.device_count()-1}")
        
        torch.cuda.set_device(gpu_idx)
        torch_device = torch.device(f'cuda:{gpu_idx}')
    
    logger.info(f"Using device: {torch_device}")
    if torch_device.type == 'cuda':
        logger.info(f"GPU: {torch.cuda.get_device_name(torch_device.index)}")
    
    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Create a dummy dataset for model loading
    dummy_dataset = FilteredCIFAR10(
        data_path=".",
        dataset_type="natural",
        train_images=train_images[:100],
        train_labels=train_labels[:100],
        test_images=test_images[:100],
        test_labels=test_labels[:100],
    )
    
    # Load the model
    logger.info("Loading model from checkpoint...")
    model, _ = model_utils.make_and_restore_model(
        arch='resnet18',
        dataset=dummy_dataset,
        resume_path=model_path,
        parallel=False,
    )
    
    # Move model to device
    model = model.to(torch_device)
    model.eval()
    
    logger.info("Model loaded successfully")
    
    # Generate adversarial examples for train set
    logger.info("\n--- Processing Train Set ---")
    train_results = generate_adversarial_for_split(
        model=model,
        images=train_images,
        labels=train_labels,
        output_dir=output_dir,
        split_name='train',
        attack_type=attack_type,
        attack_config=attack_config,
        torch_device=torch_device,
        batch_size=batch_size,
        workers=workers,
        logger=logger,
    )
    
    # Generate adversarial examples for test set
    logger.info("\n--- Processing Test Set ---")
    test_results = generate_adversarial_for_split(
        model=model,
        images=test_images,
        labels=test_labels,
        output_dir=output_dir,
        split_name='test',
        attack_type=attack_type,
        attack_config=attack_config,
        torch_device=torch_device,
        batch_size=batch_size,
        workers=workers,
        logger=logger,
    )
    
    # Save combined metadata
    metadata = {
        'attack_type': attack_type,
        'attack_config': attack_config,
        'model_path': str(model_path),
        'train': train_results,
        'test': test_results,
        'generation_date': datetime.now().isoformat(),
        'device': str(torch_device),
    }
    
    with open(output_path / "metadata.json", 'w') as f:
        json.dump(metadata, f, indent=2)
    
    np.save(output_path / "metadata.npy", metadata)
    
    logger.info(f"\nSaved adversarial dataset to {output_path}")
    logger.info(f"  Train: {train_results['num_samples']} samples, {train_results['attack_success_rate']:.2f}% attack success")
    logger.info(f"  Test: {test_results['num_samples']} samples, {test_results['attack_success_rate']:.2f}% attack success")
    
    # Clean up
    del model
    cleanup_gpu_memory()
    
    return metadata


def get_dataset_paths(model_name: str, filtered_base_dir: str, natural_dir: str) -> Tuple[str, str, str]:
    """
    Get the correct train and test data paths for a given model.
    
    Args:
        model_name: Name of the model (e.g., 'resnet18_natural', 'resnet18_high_variance_r15')
        filtered_base_dir: Base directory for filtered datasets
        natural_dir: Directory for natural CIFAR-10
        
    Returns:
        Tuple of (train_dir, test_dir, dataset_type)
        where dataset_type is 'natural' or 'filtered'
    """
    if model_name == 'resnet18_natural':
        return natural_dir, natural_dir, 'natural'
    
    # Parse model name to extract variance type and radius
    # Format: resnet18_{variance}_r{radius}
    parts = model_name.replace('resnet18_', '').split('_r')
    if len(parts) == 2:
        variance_type = parts[0]  # 'high_variance' or 'low_variance'
        radius = parts[1]  # e.g., '15'
        
        # Construct paths
        filtered_dir = Path(filtered_base_dir) / f"filtered_r{radius}"
        
        # Find the actual subdirectory names (may have .0 suffix)
        train_dirs = list(filtered_dir.glob(f"cifar10_train_{variance_type}_r{radius}*"))
        test_dirs = list(filtered_dir.glob(f"cifar10_test_{variance_type}_r{radius}*"))
        
        if train_dirs and test_dirs:
            return str(train_dirs[0]), str(test_dirs[0]), 'filtered'
    
    raise ValueError(f"Could not determine dataset paths for model: {model_name}")


def load_dataset_for_model(
    model_name: str,
    filtered_base_dir: str,
    natural_dir: str,
    logger: Optional[logging.Logger] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Load the correct train and test data for a given model.
    
    Args:
        model_name: Name of the model
        filtered_base_dir: Base directory for filtered datasets
        natural_dir: Directory for natural CIFAR-10
        logger: Optional logger
        
    Returns:
        Tuple of (train_images, train_labels, test_images, test_labels)
    """
    if logger is None:
        logger = logging.getLogger('adversarial_generation')
    
    train_dir, test_dir, dataset_type = get_dataset_paths(model_name, filtered_base_dir, natural_dir)
    
    logger.info(f"Loading dataset for {model_name}")
    logger.info(f"  Dataset type: {dataset_type}")
    logger.info(f"  Train dir: {train_dir}")
    logger.info(f"  Test dir: {test_dir}")
    
    if dataset_type == 'natural':
        train_images, train_labels, _ = load_natural_dataset(natural_dir, "train")
        test_images, test_labels, _ = load_natural_dataset(natural_dir, "test")
    else:
        train_images, train_labels, _ = load_filtered_dataset(train_dir)
        test_images, test_labels, _ = load_filtered_dataset(test_dir)
    
    logger.info(f"  Loaded: {len(train_images)} train, {len(test_images)} test images")
    
    return train_images, train_labels, test_images, test_labels


def run_adversarial_pipeline(
    models_dir: str,
    natural_dir: str,
    filtered_base_dir: str,
    output_dir: str,
    log_dir: str,
    device: str,
    attack_types: List[str] = ['pgd', 'fgsm'],
    batch_size: int = GENERATION_BATCH_SIZE,
    workers: int = 4,
    cutoff_radii: List[int] = [5, 10, 15],
    generate_for_natural: bool = True,
    generate_for_filtered: bool = True,
) -> Dict[str, Any]:
    """
    Run the complete adversarial example generation pipeline.
    
    This function generates adversarial examples for all trained models
    using both PGD and FGSM attacks. Each model uses its CORRESPONDING
    dataset (natural or filtered).
    
    Args:
        models_dir: Directory containing trained model checkpoints
        natural_dir: Path to natural CIFAR-10 dataset
        filtered_base_dir: Base path containing filtered_r{5,10,15} directories
        output_dir: Directory to save adversarial datasets
        log_dir: Directory for generation logs
        device: Device to use for generation (must be specified)
        attack_types: List of attack types to use
        batch_size: Batch size for generation
        workers: Number of data loading workers
        cutoff_radii: List of cutoff radii for filtered models
        generate_for_natural: Whether to generate for natural model
        generate_for_filtered: Whether to generate for filtered models
        
    Returns:
        Dictionary containing all generation results
    """
    # Set up main logger
    logger = setup_logging(log_dir, "adversarial_pipeline")
    
    logger.info("=" * 80)
    logger.info("ADVERSARIAL REDUNDANCY - ADVERSARIAL EXAMPLE GENERATION PIPELINE")
    logger.info("=" * 80)
    logger.info(f"Models directory: {models_dir}")
    logger.info(f"Natural dataset: {natural_dir}")
    logger.info(f"Filtered base dir: {filtered_base_dir}")
    logger.info(f"Output directory: {output_dir}")
    logger.info(f"Log directory: {log_dir}")
    logger.info(f"Device: {device}")
    logger.info(f"Attack types: {attack_types}")
    logger.info(f"Batch size: {batch_size}")
    logger.info(f"Cutoff radii: {cutoff_radii}")
    logger.info("=" * 80)
    
    # Track all results
    results = {
        'config': {
            'models_dir': models_dir,
            'natural_dir': natural_dir,
            'filtered_base_dir': filtered_base_dir,
            'output_dir': output_dir,
            'device': device,
            'attack_types': attack_types,
            'batch_size': batch_size,
            'cutoff_radii': cutoff_radii,
            'pgd_config': PGD_CONFIG,
            'fgsm_config': FGSM_CONFIG,
        },
        'generations': {},
        'start_time': datetime.now().isoformat(),
    }
    
    # Create output directory
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # Find all model checkpoints
    models_path = Path(models_dir)
    model_configs = []  # List of (model_name, checkpoint_path)
    
    if generate_for_natural:
        natural_model_dir = models_path / "resnet18_natural"
        if natural_model_dir.exists():
            checkpoint = natural_model_dir / "checkpoint.pt.best"
            if checkpoint.exists():
                model_configs.append(('resnet18_natural', str(checkpoint)))
            else:
                logger.warning(f"Checkpoint not found: {checkpoint}")
        else:
            logger.warning(f"Natural model directory not found: {natural_model_dir}")
    
    if generate_for_filtered:
        for radius in cutoff_radii:
            for variance in ['high_variance', 'low_variance']:
                model_name = f"resnet18_{variance}_r{radius}"
                model_dir = models_path / model_name
                if model_dir.exists():
                    checkpoint = model_dir / "checkpoint.pt.best"
                    if checkpoint.exists():
                        model_configs.append((model_name, str(checkpoint)))
                    else:
                        logger.warning(f"Checkpoint not found: {checkpoint}")
                else:
                    logger.warning(f"Model directory not found: {model_dir}")
    
    logger.info(f"Found {len(model_configs)} model checkpoints")
    
    # Generate adversarial examples for each model and attack type
    for model_name, checkpoint_path in model_configs:
        # Load the correct dataset for this model
        try:
            train_images, train_labels, test_images, test_labels = load_dataset_for_model(
                model_name, filtered_base_dir, natural_dir, logger
            )
        except Exception as e:
            logger.error(f"Failed to load dataset for {model_name}: {e}")
            for attack_type in attack_types:
                results['generations'][f"{model_name}_{attack_type}"] = {'error': str(e)}
            continue
        
        for attack_type in attack_types:
            logger.info("\n" + "=" * 60)
            logger.info(f"Processing: {model_name} with {attack_type.upper()} attack")
            logger.info("=" * 60)
            
            output_subdir = Path(output_dir) / model_name / attack_type
            
            try:
                gen_results = generate_adversarial_dataset(
                    model_path=checkpoint_path,
                    train_images=train_images,
                    train_labels=train_labels,
                    test_images=test_images,
                    test_labels=test_labels,
                    output_dir=str(output_subdir),
                    attack_type=attack_type,
                    device=device,
                    batch_size=batch_size,
                    workers=workers,
                    logger=logger,
                )
                
                results['generations'][f"{model_name}_{attack_type}"] = gen_results
                logger.info(f"✓ {model_name} {attack_type.upper()} generation complete")
                
            except Exception as e:
                logger.error(f"Failed to generate {attack_type} examples for {model_name}: {e}")
                import traceback
                logger.error(traceback.format_exc())
                results['generations'][f"{model_name}_{attack_type}"] = {'error': str(e)}
    
    # Save final results
    results['end_time'] = datetime.now().isoformat()
    
    results_path = Path(output_dir) / "adversarial_generation_results.json"
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    logger.info("\n" + "=" * 80)
    logger.info("ADVERSARIAL GENERATION PIPELINE COMPLETE")
    logger.info("=" * 80)
    logger.info(f"Results saved to: {results_path}")
    logger.info(f"Generations completed: {len([k for k, v in results['generations'].items() if 'error' not in v])}")
    logger.info(f"Generations failed: {len([k for k, v in results['generations'].items() if 'error' in v])}")
    
    for name, info in results['generations'].items():
        if 'error' in info:
            logger.info(f"  ✗ {name}: FAILED - {info['error']}")
        else:
            train_success = info.get('train', {}).get('attack_success_rate', 'N/A')
            test_success = info.get('test', {}).get('attack_success_rate', 'N/A')
            logger.info(f"  ✓ {name}: train={train_success:.2f}%, test={test_success:.2f}% attack success")
    
    return results


def main():
    """
    Main entry point for adversarial example generation.
    """
    parser = argparse.ArgumentParser(
        description="Generate adversarial examples using the robustness library",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Required arguments
    parser.add_argument(
        "--model-path",
        type=str,
        required=True,
        help="Path to the model checkpoint (e.g., ./models/resnet18_natural/checkpoint.pt.best)"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        required=True,
        help="Directory to save adversarial dataset"
    )
    parser.add_argument(
        "--device",
        type=str,
        required=True,
        help="Device to use (e.g., 'cuda:0', 'cuda:1', '0', '1'). This is mandatory for GPU control."
    )
    
    # Dataset paths
    parser.add_argument(
        "--train-data-dir",
        type=str,
        required=True,
        help="Path to training data directory"
    )
    parser.add_argument(
        "--test-data-dir",
        type=str,
        required=True,
        help="Path to test data directory"
    )
    parser.add_argument(
        "--dataset-type",
        type=str,
        choices=['natural', 'filtered'],
        required=True,
        help="Type of dataset ('natural' or 'filtered')"
    )
    
    # Attack configuration
    parser.add_argument(
        "--attack-type",
        type=str,
        choices=['pgd', 'fgsm'],
        default='pgd',
        help="Type of attack to use"
    )
    
    # Generation parameters
    parser.add_argument(
        "--batch-size",
        type=int,
        default=GENERATION_BATCH_SIZE,
        help="Batch size for adversarial generation"
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=4,
        help="Number of data loading workers"
    )
    
    # Logging
    parser.add_argument(
        "--log-dir",
        type=str,
        default="./logs/adversarial",
        help="Directory for generation logs"
    )
    
    args = parser.parse_args()
    
    # Set up logging
    logger = setup_logging(args.log_dir, f"adversarial_{args.attack_type}")
    
    # Load data based on dataset type
    logger.info(f"Loading data from {args.train_data_dir} and {args.test_data_dir}...")
    
    if args.dataset_type == 'natural':
        train_images, train_labels, _ = load_natural_dataset(args.train_data_dir, "train")
        test_images, test_labels, _ = load_natural_dataset(args.test_data_dir, "test")
    else:
        train_images, train_labels, _ = load_filtered_dataset(args.train_data_dir)
        test_images, test_labels, _ = load_filtered_dataset(args.test_data_dir)
    
    logger.info(f"Loaded {len(train_images)} train and {len(test_images)} test images")
    
    # Generate adversarial examples
    results = generate_adversarial_dataset(
        model_path=args.model_path,
        train_images=train_images,
        train_labels=train_labels,
        test_images=test_images,
        test_labels=test_labels,
        output_dir=args.output_dir,
        attack_type=args.attack_type,
        device=args.device,
        batch_size=args.batch_size,
        workers=args.workers,
        logger=logger,
    )
    
    logger.info("Done!")
    
    # Print summary
    print("\n" + "=" * 60)
    print("ADVERSARIAL EXAMPLE GENERATION COMPLETE")
    print("=" * 60)
    print(f"Attack type: {results['attack_type']}")
    print(f"Train samples: {results['train']['num_samples']}")
    print(f"  Train clean accuracy: {results['train']['clean_accuracy']:.2f}%")
    print(f"  Train adversarial accuracy: {results['train']['adversarial_accuracy']:.2f}%")
    print(f"  Train attack success rate: {results['train']['attack_success_rate']:.2f}%")
    print(f"Test samples: {results['test']['num_samples']}")
    print(f"  Test clean accuracy: {results['test']['clean_accuracy']:.2f}%")
    print(f"  Test adversarial accuracy: {results['test']['adversarial_accuracy']:.2f}%")
    print(f"  Test attack success rate: {results['test']['attack_success_rate']:.2f}%")
    print(f"Output directory: {args.output_dir}")
    print("=" * 60)


if __name__ == "__main__":
    main()
