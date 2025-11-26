#!/usr/bin/env python3
"""
Training pipeline for ResNet18 models using the robustness library.

This script trains ResNet18 models on:
1. Natural CIFAR-10 dataset
2. High-variance (low-pass filtered) datasets for r=5, 10, 15
3. Low-variance (high-pass filtered) datasets for r=5, 10, 15

The robustness library is used to facilitate later adversarial example generation.

Usage:
    python train_models.py --natural-dir ./data/cifar10_natural \
                           --filtered-base-dir ./data \
                           --output-dir ./models \
                           --log-dir ./logs/training \
                           --epochs 100
"""

import os
import sys
import gc
import argparse
import logging
import json
from pathlib import Path
from datetime import datetime
from typing import Tuple, Optional, Dict, Any

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from robustness import model_utils, train, defaults, datasets as robustness_datasets
from robustness.tools import helpers
from cox.utils import Parameters

from train.datasets import FilteredCIFAR10, load_filtered_dataset, load_natural_dataset


def setup_logging(log_dir: str, run_name: str) -> logging.Logger:
    """
    Set up logging for training.
    
    Args:
        log_dir: Directory for log files
        run_name: Name of the training run
        
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
    Clean up GPU memory after training.
    
    This function ensures that model weights are removed from GPU memory
    to prevent out-of-memory errors during sequential training runs.
    """
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()


def get_training_args(
    output_dir: str,
    epochs: int = 100,
    lr: float = 0.1,
    batch_size: int = 128,
    weight_decay: float = 5e-4,
    momentum: float = 0.9,
    step_lr: int = 50,
    step_lr_gamma: float = 0.1,
    workers: int = 4,
    log_iters: int = 100,
    save_ckpt_iters: int = -1,
    adv_train: int = 0,
):
    """
    Create training arguments for the robustness library.
    
    Args:
        output_dir: Directory to save checkpoints
        epochs: Number of training epochs
        lr: Initial learning rate
        batch_size: Training batch size
        weight_decay: Weight decay for optimizer
        momentum: Momentum for SGD optimizer
        step_lr: Epoch to reduce learning rate
        step_lr_gamma: Factor to reduce learning rate by
        workers: Number of data loading workers
        log_iters: How often to log training progress
        save_ckpt_iters: How often to save checkpoints (-1 for end only)
        adv_train: Whether to do adversarial training (0 for standard)
        
    Returns:
        Training parameters object
    """
    train_kwargs = {
        'out_dir': output_dir,
        'epochs': epochs,
        'lr': lr,
        'batch_size': batch_size,
        'weight_decay': weight_decay,
        'momentum': momentum,
        'step_lr': step_lr,
        'step_lr_gamma': step_lr_gamma,
        'workers': workers,
        'log_iters': log_iters,
        'save_ckpt_iters': save_ckpt_iters,
        'adv_train': adv_train,
        'adv_eval': 0,
    }
    
    # Create Parameters object and fill in defaults
    args = Parameters(train_kwargs)
    args = defaults.check_and_fill_args(
        args,
        defaults.TRAINING_ARGS,
        robustness_datasets.CIFAR
    )
    
    return args


def train_single_model(
    dataset: FilteredCIFAR10,
    output_dir: str,
    model_name: str,
    epochs: int = 100,
    lr: float = 0.1,
    batch_size: int = 128,
    workers: int = 4,
    log_iters: int = 100,
    logger: Optional[logging.Logger] = None,
    resume_path: Optional[str] = None,
    device: Optional[str] = None,
) -> Tuple[str, Dict[str, Any]]:
    """
    Train a single ResNet18 model on a dataset.
    
    This function trains a ResNet18 model using the robustness library's
    training pipeline. After training, it cleans up GPU memory.
    
    Args:
        dataset: FilteredCIFAR10 dataset to train on
        output_dir: Directory to save model checkpoints
        model_name: Name for the model (used in file names)
        epochs: Number of training epochs
        lr: Initial learning rate
        batch_size: Training batch size
        workers: Number of data loading workers
        log_iters: How often to log training progress
        logger: Optional logger for verbose output
        resume_path: Optional path to resume from checkpoint
        device: Device to use for training (e.g., "cuda:0", "cuda:1", "cpu", or "0", "1" for GPU index)
        
    Returns:
        Tuple of (checkpoint_path, training_stats)
    """
    if logger is None:
        logger = logging.getLogger(model_name)
    
    # Create output directory
    model_output_dir = Path(output_dir) / model_name
    model_output_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"=" * 60)
    logger.info(f"Training model: {model_name}")
    logger.info(f"Output directory: {model_output_dir}")
    logger.info(f"Epochs: {epochs}, LR: {lr}, Batch size: {batch_size}")
    logger.info(f"=" * 60)
    
    # Determine device
    if device is None:
        # Default: use first available GPU or CPU
        if torch.cuda.is_available():
            device_str = 'cuda:0'
        else:
            device_str = 'cpu'
    elif device.isdigit():
        # If device is just a number, interpret as GPU index
        device_str = f'cuda:{device}'
    else:
        # Use device string as-is (e.g., "cuda:0", "cuda:1", "cpu")
        device_str = device
    
    torch_device = torch.device(device_str)
    logger.info(f"Using device: {torch_device}")
    
    # Validate device is available
    if torch_device.type == 'cuda':
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is not available but cuda device was requested")
        if torch_device.index is not None and torch_device.index >= torch.cuda.device_count():
            raise RuntimeError(f"GPU {torch_device.index} is not available. Available GPUs: 0-{torch.cuda.device_count()-1}")
    
    # Create model using robustness library
    logger.info("Creating ResNet18 model...")
    model, _ = model_utils.make_and_restore_model(
        arch='resnet18',
        dataset=dataset,
        resume_path=resume_path,
    )
    
    # Move model to device
    model = model.to(torch_device)
    
    if torch_device.type == 'cuda':
        gpu_idx = torch_device.index if torch_device.index is not None else 0
        logger.info(f"GPU: {torch.cuda.get_device_name(gpu_idx)}")
        logger.info(f"GPU Memory: {torch.cuda.get_device_properties(gpu_idx).total_memory / 1e9:.2f} GB")
    
    # Get training arguments
    args = get_training_args(
        output_dir=str(model_output_dir),
        epochs=epochs,
        lr=lr,
        batch_size=batch_size,
        workers=workers,
        log_iters=log_iters,
    )
    
    # Create data loaders
    logger.info("Creating data loaders...")
    train_loader, val_loader = dataset.make_loaders(
        batch_size=batch_size,
        workers=workers,
        data_aug=True,
    )
    logger.info(f"Train batches: {len(train_loader)}, Val batches: {len(val_loader)}")
    
    # Train the model
    logger.info("Starting training...")
    start_time = datetime.now()
    
    try:
        model = train.train_model(
            args,
            model,
            (train_loader, val_loader),
            store=None,  # Don't use cox store
        )
    except Exception as e:
        logger.error(f"Training failed: {e}")
        raise
    
    end_time = datetime.now()
    training_time = (end_time - start_time).total_seconds()
    
    logger.info(f"Training completed in {training_time:.2f} seconds")
    
    # Save final checkpoint path
    checkpoint_path = model_output_dir / "checkpoint.pt"
    
    # Collect training stats
    training_stats = {
        'model_name': model_name,
        'epochs': epochs,
        'lr': lr,
        'batch_size': batch_size,
        'training_time_seconds': training_time,
        'checkpoint_path': str(checkpoint_path),
        'device': str(torch_device),
    }
    
    # Save training stats
    stats_path = model_output_dir / "training_stats.json"
    with open(stats_path, 'w') as f:
        json.dump(training_stats, f, indent=2)
    logger.info(f"Saved training stats to {stats_path}")
    
    # Clean up GPU memory
    logger.info("Cleaning up GPU memory...")
    del model
    del train_loader
    del val_loader
    cleanup_gpu_memory()
    
    if torch_device.type == 'cuda':
        logger.info(f"GPU memory after cleanup: {torch.cuda.memory_allocated() / 1e6:.2f} MB allocated")
    
    logger.info(f"Model saved to: {checkpoint_path}")
    
    return str(checkpoint_path), training_stats


def run_training_pipeline(
    natural_dir: str,
    filtered_base_dir: str,
    output_dir: str,
    log_dir: str,
    epochs: int = 100,
    lr: float = 0.1,
    batch_size: int = 128,
    workers: int = 4,
    log_iters: int = 100,
    cutoff_radii: list = [5, 10, 15],
    train_natural: bool = True,
    train_filtered: bool = True,
    device: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Run the complete training pipeline.
    
    This function trains ResNet18 models on:
    1. Natural CIFAR-10 dataset (if train_natural=True)
    2. High-variance datasets for each cutoff radius (if train_filtered=True)
    3. Low-variance datasets for each cutoff radius (if train_filtered=True)
    
    Args:
        natural_dir: Path to natural CIFAR-10 dataset
        filtered_base_dir: Base path containing filtered_r{5,10,15} directories
        output_dir: Directory to save model checkpoints
        log_dir: Directory for training logs
        epochs: Number of training epochs
        lr: Initial learning rate
        batch_size: Training batch size
        workers: Number of data loading workers
        log_iters: How often to log training progress
        cutoff_radii: List of cutoff radii to train on
        train_natural: Whether to train on natural dataset
        train_filtered: Whether to train on filtered datasets
        device: Device to use for training (e.g., "cuda:0", "cuda:1", "cpu", or "0", "1" for GPU index)
        
    Returns:
        Dictionary containing all training results
    """
    # Set up main logger
    logger = setup_logging(log_dir, "training_pipeline")
    
    logger.info("=" * 80)
    logger.info("ADVERSARIAL REDUNDANCY - TRAINING PIPELINE")
    logger.info("=" * 80)
    logger.info(f"Natural dataset: {natural_dir}")
    logger.info(f"Filtered base dir: {filtered_base_dir}")
    logger.info(f"Output directory: {output_dir}")
    logger.info(f"Log directory: {log_dir}")
    logger.info(f"Epochs: {epochs}, LR: {lr}, Batch size: {batch_size}")
    logger.info(f"Cutoff radii: {cutoff_radii}")
    logger.info(f"Device: {device if device else 'auto (cuda:0 or cpu)'}")
    logger.info("=" * 80)
    
    # Track all results
    results = {
        'config': {
            'natural_dir': natural_dir,
            'filtered_base_dir': filtered_base_dir,
            'output_dir': output_dir,
            'epochs': epochs,
            'lr': lr,
            'batch_size': batch_size,
            'cutoff_radii': cutoff_radii,
        },
        'models': {},
        'start_time': datetime.now().isoformat(),
    }
    
    # Create output directory
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # =========================================================================
    # Train on Natural Dataset
    # =========================================================================
    if train_natural:
        logger.info("\n" + "=" * 60)
        logger.info("TRAINING ON NATURAL DATASET")
        logger.info("=" * 60)
        
        try:
            # Load natural dataset
            train_images, train_labels, _ = load_natural_dataset(natural_dir, "train")
            test_images, test_labels, _ = load_natural_dataset(natural_dir, "test")
            
            logger.info(f"Loaded natural dataset: {len(train_images)} train, {len(test_images)} test")
            
            # Create dataset
            dataset = FilteredCIFAR10(
                data_path=natural_dir,
                dataset_type="natural",
                train_images=train_images,
                train_labels=train_labels,
                test_images=test_images,
                test_labels=test_labels,
            )
            
            # Train model
            checkpoint_path, stats = train_single_model(
                dataset=dataset,
                output_dir=output_dir,
                model_name="resnet18_natural",
                epochs=epochs,
                lr=lr,
                batch_size=batch_size,
                workers=workers,
                log_iters=log_iters,
                logger=logger,
                device=device,
            )
            
            results['models']['natural'] = {
                'checkpoint_path': checkpoint_path,
                'stats': stats,
            }
            
            logger.info(f"✓ Natural model training complete: {checkpoint_path}")
            
        except Exception as e:
            logger.error(f"Failed to train natural model: {e}")
            results['models']['natural'] = {'error': str(e)}
    
    # =========================================================================
    # Train on Filtered Datasets
    # =========================================================================
    if train_filtered:
        for radius in cutoff_radii:
            logger.info("\n" + "=" * 60)
            logger.info(f"TRAINING ON FILTERED DATASETS (r={radius})")
            logger.info("=" * 60)
            
            # Determine the filtered directory path
            # Handle both integer and float naming conventions
            filtered_dir = Path(filtered_base_dir) / f"filtered_r{radius}"
            if not filtered_dir.exists():
                logger.warning(f"Filtered directory not found: {filtered_dir}")
                continue
            
            # Find the actual subdirectory names (may have .0 suffix)
            train_high_var_dirs = list(filtered_dir.glob(f"cifar10_train_high_variance_r{radius}*"))
            train_low_var_dirs = list(filtered_dir.glob(f"cifar10_train_low_variance_r{radius}*"))
            test_high_var_dirs = list(filtered_dir.glob(f"cifar10_test_high_variance_r{radius}*"))
            test_low_var_dirs = list(filtered_dir.glob(f"cifar10_test_low_variance_r{radius}*"))
            
            # -----------------------------------------------------------------
            # Train High-Variance Model
            # -----------------------------------------------------------------
            if train_high_var_dirs and test_high_var_dirs:
                train_high_var_dir = train_high_var_dirs[0]
                test_high_var_dir = test_high_var_dirs[0]
                
                logger.info(f"\nTraining high-variance model (r={radius})...")
                logger.info(f"Train dir: {train_high_var_dir}")
                logger.info(f"Test dir: {test_high_var_dir}")
                
                try:
                    # Load filtered data
                    train_images, train_labels, _ = load_filtered_dataset(str(train_high_var_dir))
                    test_images, test_labels, _ = load_filtered_dataset(str(test_high_var_dir))
                    
                    logger.info(f"Loaded: {len(train_images)} train, {len(test_images)} test")
                    
                    # Create dataset
                    dataset = FilteredCIFAR10(
                        data_path=str(train_high_var_dir),
                        dataset_type=f"high_variance_r{radius}",
                        train_images=train_images,
                        train_labels=train_labels,
                        test_images=test_images,
                        test_labels=test_labels,
                    )
                    
                    # Train model
                    checkpoint_path, stats = train_single_model(
                        dataset=dataset,
                        output_dir=output_dir,
                        model_name=f"resnet18_high_variance_r{radius}",
                        epochs=epochs,
                        lr=lr,
                        batch_size=batch_size,
                        workers=workers,
                        log_iters=log_iters,
                        logger=logger,
                        device=device,
                    )
                    
                    results['models'][f'high_variance_r{radius}'] = {
                        'checkpoint_path': checkpoint_path,
                        'stats': stats,
                    }
                    
                    logger.info(f"✓ High-variance r={radius} training complete")
                    
                except Exception as e:
                    logger.error(f"Failed to train high-variance r={radius}: {e}")
                    results['models'][f'high_variance_r{radius}'] = {'error': str(e)}
            else:
                logger.warning(f"High-variance directories not found for r={radius}")
            
            # -----------------------------------------------------------------
            # Train Low-Variance Model
            # -----------------------------------------------------------------
            if train_low_var_dirs and test_low_var_dirs:
                train_low_var_dir = train_low_var_dirs[0]
                test_low_var_dir = test_low_var_dirs[0]
                
                logger.info(f"\nTraining low-variance model (r={radius})...")
                logger.info(f"Train dir: {train_low_var_dir}")
                logger.info(f"Test dir: {test_low_var_dir}")
                
                try:
                    # Load filtered data
                    train_images, train_labels, _ = load_filtered_dataset(str(train_low_var_dir))
                    test_images, test_labels, _ = load_filtered_dataset(str(test_low_var_dir))
                    
                    logger.info(f"Loaded: {len(train_images)} train, {len(test_images)} test")
                    
                    # Create dataset
                    dataset = FilteredCIFAR10(
                        data_path=str(train_low_var_dir),
                        dataset_type=f"low_variance_r{radius}",
                        train_images=train_images,
                        train_labels=train_labels,
                        test_images=test_images,
                        test_labels=test_labels,
                    )
                    
                    # Train model
                    checkpoint_path, stats = train_single_model(
                        dataset=dataset,
                        output_dir=output_dir,
                        model_name=f"resnet18_low_variance_r{radius}",
                        epochs=epochs,
                        lr=lr,
                        batch_size=batch_size,
                        workers=workers,
                        log_iters=log_iters,
                        logger=logger,
                        device=device,
                    )
                    
                    results['models'][f'low_variance_r{radius}'] = {
                        'checkpoint_path': checkpoint_path,
                        'stats': stats,
                    }
                    
                    logger.info(f"✓ Low-variance r={radius} training complete")
                    
                except Exception as e:
                    logger.error(f"Failed to train low-variance r={radius}: {e}")
                    results['models'][f'low_variance_r{radius}'] = {'error': str(e)}
            else:
                logger.warning(f"Low-variance directories not found for r={radius}")
    
    # =========================================================================
    # Save Final Results
    # =========================================================================
    results['end_time'] = datetime.now().isoformat()
    
    results_path = Path(output_dir) / "training_results.json"
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    logger.info("\n" + "=" * 80)
    logger.info("TRAINING PIPELINE COMPLETE")
    logger.info("=" * 80)
    logger.info(f"Results saved to: {results_path}")
    logger.info(f"Models trained: {len(results['models'])}")
    for name, info in results['models'].items():
        if 'error' in info:
            logger.info(f"  ✗ {name}: FAILED - {info['error']}")
        else:
            logger.info(f"  ✓ {name}: {info['checkpoint_path']}")
    
    return results


def main():
    """
    Main entry point for the training pipeline.
    """
    parser = argparse.ArgumentParser(
        description="Train ResNet18 models on natural and filtered CIFAR-10 datasets",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Data paths
    parser.add_argument(
        "--natural-dir",
        type=str,
        default="./data/cifar10_natural",
        help="Path to natural CIFAR-10 dataset"
    )
    parser.add_argument(
        "--filtered-base-dir",
        type=str,
        default="./data",
        help="Base path containing filtered_r{5,10,15} directories"
    )
    
    # Output paths
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./models",
        help="Directory to save model checkpoints"
    )
    parser.add_argument(
        "--log-dir",
        type=str,
        default="./logs/training",
        help="Directory for training logs"
    )
    
    # Training hyperparameters
    parser.add_argument(
        "--epochs",
        type=int,
        default=100,
        help="Number of training epochs"
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=0.1,
        help="Initial learning rate"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=128,
        help="Training batch size"
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=4,
        help="Number of data loading workers"
    )
    parser.add_argument(
        "--log-iters",
        type=int,
        default=100,
        help="How often to log training progress"
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device to use for training (e.g., 'cuda:0', 'cuda:1', 'cpu', or '0', '1' for GPU index). Default: auto-select first available GPU"
    )
    
    # Dataset selection
    parser.add_argument(
        "--cutoff-radii",
        type=int,
        nargs="+",
        default=[5, 10, 15],
        help="Cutoff radii for filtered datasets"
    )
    parser.add_argument(
        "--no-natural",
        action="store_true",
        help="Skip training on natural dataset"
    )
    parser.add_argument(
        "--no-filtered",
        action="store_true",
        help="Skip training on filtered datasets"
    )
    parser.add_argument(
        "--natural-only",
        action="store_true",
        help="Only train on natural dataset"
    )
    parser.add_argument(
        "--filtered-only",
        action="store_true",
        help="Only train on filtered datasets"
    )
    
    args = parser.parse_args()
    
    # Determine what to train
    train_natural = not args.no_natural and not args.filtered_only
    train_filtered = not args.no_filtered and not args.natural_only
    
    # Run the pipeline
    results = run_training_pipeline(
        natural_dir=args.natural_dir,
        filtered_base_dir=args.filtered_base_dir,
        output_dir=args.output_dir,
        log_dir=args.log_dir,
        epochs=args.epochs,
        lr=args.lr,
        batch_size=args.batch_size,
        workers=args.workers,
        log_iters=args.log_iters,
        cutoff_radii=args.cutoff_radii,
        train_natural=train_natural,
        train_filtered=train_filtered,
        device=args.device,
    )
    
    # Exit with error code if any training failed
    failed = sum(1 for info in results['models'].values() if 'error' in info)
    if failed > 0:
        print(f"\n{failed} model(s) failed to train. Check logs for details.")
        sys.exit(1)


if __name__ == "__main__":
    main()

