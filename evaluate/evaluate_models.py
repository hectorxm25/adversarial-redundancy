#!/usr/bin/env python3
"""
Evaluation pipeline for trained ResNet18 models.

This script evaluates trained models on test datasets to compute accuracy metrics.
It can evaluate models on:
1. Natural CIFAR-10 test set
2. Filtered (high-variance and low-variance) test sets

Usage:
    python evaluate_models.py --models-dir ./models \
                              --natural-dir ./data/cifar10_natural \
                              --filtered-base-dir ./data \
                              --output-dir ./evaluation_results
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

# IMPORTANT: Set CUDA_VISIBLE_DEVICES before importing torch if device is specified
# This must be done via environment variable or command-line argument
# We'll handle it in main() function, but need to be careful about import order

import numpy as np

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
from torch.utils.data import DataLoader

from robustness import model_utils, train as robustness_train, defaults
from robustness.tools import helpers
from cox.utils import Parameters

from train.datasets import FilteredCIFAR10, load_filtered_dataset, load_natural_dataset


def setup_logging(log_dir: str, run_name: str) -> logging.Logger:
    """
    Set up logging for evaluation.
    
    Args:
        log_dir: Directory for log files
        run_name: Name of the evaluation run
        
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


def find_checkpoint_path(model_dir: Path) -> Optional[Path]:
    """
    Find the checkpoint file in a model directory.
    
    Checks for checkpoint.pt.best first, then falls back to checkpoint.pt.
    
    Args:
        model_dir: Path to the model directory
        
    Returns:
        Path to the checkpoint file if found, None otherwise
    """
    # Try checkpoint.pt.best first (preferred)
    checkpoint_best = model_dir / "checkpoint.pt.best"
    if checkpoint_best.exists():
        return checkpoint_best
    
    # Fall back to checkpoint.pt
    checkpoint_regular = model_dir / "checkpoint.pt"
    if checkpoint_regular.exists():
        return checkpoint_regular
    
    return None


def cleanup_gpu_memory():
    """
    Clean up GPU memory after evaluation.
    """
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()


def get_evaluation_args(
    adv_eval: int = 0,
) -> Parameters:
    """
    Create evaluation arguments for the robustness library.
    
    Args:
        adv_eval: Whether to do adversarial evaluation (0 for natural, 1 for adversarial)
        
    Returns:
        Evaluation parameters object
    """
    eval_kwargs = {
        'adv_eval': adv_eval,
    }
    
    args = Parameters(eval_kwargs)
    
    # Fill in defaults required by eval_model
    if not hasattr(args, 'adv_eval'):
        args.adv_eval = adv_eval
    
    return args


def evaluate_model(
    checkpoint_path: str,
    dataset: FilteredCIFAR10,
    model_name: str,
    batch_size: int = 128,
    workers: int = 4,
    device: Optional[str] = None,
    adv_eval: int = 0,
    logger: Optional[logging.Logger] = None,
) -> Dict[str, Any]:
    """
    Evaluate a trained model on a test dataset.
    
    Args:
        checkpoint_path: Path to the model checkpoint
        dataset: FilteredCIFAR10 dataset to evaluate on
        model_name: Name for the model (used in logging)
        batch_size: Evaluation batch size
        workers: Number of data loading workers
        device: Device to use for evaluation (e.g., "cuda:0", "cuda:1", "cpu", or "0", "1" for GPU index)
        adv_eval: Whether to do adversarial evaluation (0 for natural, 1 for adversarial)
        logger: Optional logger for verbose output
        
    Returns:
        Dictionary containing evaluation results (accuracy, loss, etc.)
    """
    if logger is None:
        logger = logging.getLogger(model_name)
    
    logger.info(f"=" * 60)
    logger.info(f"Evaluating model: {model_name}")
    logger.info(f"Checkpoint: {checkpoint_path}")
    logger.info(f"Adversarial evaluation: {adv_eval}")
    logger.info(f"=" * 60)
    
    # Determine device
    # Note: After CUDA_VISIBLE_DEVICES is set, the specified GPU appears as cuda:0
    # So we always use cuda:0 here, since DataParallel expects cuda:0
    if device is None:
        if torch.cuda.is_available():
            device_str = 'cuda:0'
        else:
            device_str = 'cpu'
    elif device == 'cpu':
        device_str = 'cpu'
    else:
        # For CUDA devices, always use cuda:0 since CUDA_VISIBLE_DEVICES makes
        # the specified GPU appear as cuda:0
        device_str = 'cuda:0'
    
    torch_device = torch.device(device_str)
    logger.info(f"Using device: {torch_device}")
    
    # Validate device is available
    if torch_device.type == 'cuda':
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is not available but cuda device was requested")
        # After CUDA_VISIBLE_DEVICES is set, we should only see one GPU (at index 0)
        if torch.cuda.device_count() == 0:
            raise RuntimeError("No CUDA devices available")
    
    # Load the trained model
    logger.info("Loading model from checkpoint...")
    
    # The robustness library's make_and_restore_model may load the checkpoint
    # with parameters on a different device. We need to ensure everything is on cuda:0
    # after loading, since DataParallel requires cuda:0
    
    # The robustness library's make_and_restore_model will automatically call model.cuda()
    # at the end, which moves the model to the default CUDA device.
    # After CUDA_VISIBLE_DEVICES is set in the bash script, the specified GPU appears as cuda:0.
    # So the robustness library should move it to cuda:0 automatically.
    
    model, checkpoint = model_utils.make_and_restore_model(
        arch='resnet18',
        dataset=dataset,
        resume_path=checkpoint_path,
        parallel=False,  # Don't wrap in DataParallel yet - eval_model will do that
    )
    
    # After make_and_restore_model, the model should be on CUDA (cuda:0 after CUDA_VISIBLE_DEVICES).
    # However, we need to ensure all parameters are on cuda:0 for DataParallel compatibility.
    # Sometimes checkpoint loading can leave some parameters on the wrong device.
    
    if torch_device.type == 'cuda':
        # The robustness library should have moved the model to cuda:0 already
        # But we need to ensure ALL parameters and buffers are on cuda:0
        logger.info("Verifying all model parameters are on cuda:0...")
        
        # Move model to CPU first to reset any device assignments
        model = model.cpu()
        # Then explicitly move to cuda:0
        model = model.cuda(0)
        
        # Explicitly check and move all parameters and buffers to cuda:0
        moved_params = 0
        moved_buffers = 0
        for name, param in model.named_parameters():
            if param.device != torch.device('cuda:0'):
                logger.warning(f"Parameter {name} is on {param.device}, moving to cuda:0")
                param.data = param.data.to('cuda:0')
                moved_params += 1
        for name, buffer in model.named_buffers():
            if buffer.device != torch.device('cuda:0'):
                logger.warning(f"Buffer {name} is on {buffer.device}, moving to cuda:0")
                buffer.data = buffer.data.to('cuda:0')
                moved_buffers += 1
        
        if moved_params > 0 or moved_buffers > 0:
            logger.info(f"Fixed {moved_params} parameters and {moved_buffers} buffers")
        
        # Final verification - all parameters must be on cuda:0
        wrong_device_items = []
        for name, param in model.named_parameters():
            if param.device != torch.device('cuda:0'):
                wrong_device_items.append(f"{name} on {param.device}")
        for name, buffer in model.named_buffers():
            if buffer.device != torch.device('cuda:0'):
                wrong_device_items.append(f"{name} on {buffer.device}")
        
        if wrong_device_items:
            error_msg = "Failed to move all parameters/buffers to cuda:0:\n"
            error_msg += "\n".join(f"  - {item}" for item in wrong_device_items[:10])
            if len(wrong_device_items) > 10:
                error_msg += f"\n  ... and {len(wrong_device_items) - 10} more"
            raise RuntimeError(error_msg)
        
        logger.info("✓ All parameters and buffers verified on cuda:0")
    else:
        model = model.to(torch_device)
    
    model.eval()
    
    if torch_device.type == 'cuda':
        # After CUDA_VISIBLE_DEVICES, only cuda:0 is visible
        logger.info(f"GPU: {torch.cuda.get_device_name(0)}")
    
    # Create test dataloader
    logger.info("Creating test dataloader...")
    _, test_loader = dataset.make_loaders(
        batch_size=batch_size,
        workers=workers,
        data_aug=False,  # No augmentation for evaluation
        shuffle_val=False,  # Don't shuffle test set
    )
    logger.info(f"Test batches: {len(test_loader)}")
    
    # Set up evaluation arguments
    eval_args = get_evaluation_args(adv_eval=adv_eval)
    
    # Evaluate the model
    logger.info("Running evaluation...")
    start_time = datetime.now()
    
    try:
        eval_info = robustness_train.eval_model(
            args=eval_args,
            model=model,
            loader=test_loader,
            store=None,  # Don't use cox store
        )
    except Exception as e:
        logger.error(f"Evaluation failed: {e}")
        raise
    
    end_time = datetime.now()
    eval_time = (end_time - start_time).total_seconds()
    
    logger.info(f"Evaluation completed in {eval_time:.2f} seconds")
    
    # Extract results
    results = {
        'model_name': model_name,
        'checkpoint_path': checkpoint_path,
        'natural_accuracy': float(eval_info.get('nat_prec1', float('nan'))),
        'natural_loss': float(eval_info.get('nat_loss', float('nan'))),
        'adversarial_accuracy': float(eval_info.get('adv_prec1', float('nan'))) if adv_eval else None,
        'adversarial_loss': float(eval_info.get('adv_loss', float('nan'))) if adv_eval else None,
        'evaluation_time_seconds': eval_time,
        'device': str(torch_device),
        'batch_size': batch_size,
    }
    
    logger.info(f"Natural Accuracy: {results['natural_accuracy']:.2f}%")
    logger.info(f"Natural Loss: {results['natural_loss']:.4f}")
    if adv_eval and results['adversarial_accuracy'] is not None:
        logger.info(f"Adversarial Accuracy: {results['adversarial_accuracy']:.2f}%")
        logger.info(f"Adversarial Loss: {results['adversarial_loss']:.4f}")
    
    # Clean up GPU memory
    logger.info("Cleaning up GPU memory...")
    del model
    del test_loader
    cleanup_gpu_memory()
    
    if torch_device.type == 'cuda':
        logger.info(f"GPU memory after cleanup: {torch.cuda.memory_allocated() / 1e6:.2f} MB allocated")
    
    return results


def evaluate_all_models(
    models_dir: str,
    natural_dir: str,
    filtered_base_dir: str,
    output_dir: str,
    log_dir: str,
    batch_size: int = 128,
    workers: int = 4,
    device: Optional[str] = None,
    cutoff_radii: List[int] = [5, 10, 15],
    eval_natural: bool = True,
    eval_filtered: bool = True,
    eval_on_natural_test: bool = True,
    eval_on_filtered_test: bool = True,
    adv_eval: int = 0,
) -> Dict[str, Any]:
    """
    Evaluate all trained models on test datasets.
    
    This function evaluates models on:
    1. Natural test set (if eval_on_natural_test=True)
    2. Filtered test sets (if eval_on_filtered_test=True)
    
    Args:
        models_dir: Directory containing trained model checkpoints
        natural_dir: Path to natural CIFAR-10 dataset
        filtered_base_dir: Base path containing filtered_r{5,10,15} directories
        output_dir: Directory to save evaluation results
        log_dir: Directory for evaluation logs
        batch_size: Evaluation batch size
        workers: Number of data loading workers
        device: Device to use for evaluation
        cutoff_radii: List of cutoff radii to evaluate
        eval_natural: Whether to evaluate natural model
        eval_filtered: Whether to evaluate filtered models
        eval_on_natural_test: Whether to evaluate on natural test set
        eval_on_filtered_test: Whether to evaluate on filtered test sets
        adv_eval: Whether to do adversarial evaluation (0 for natural, 1 for adversarial)
        
    Returns:
        Dictionary containing all evaluation results
    """
    # Set up main logger
    logger = setup_logging(log_dir, "evaluation_pipeline")
    
    logger.info("=" * 80)
    logger.info("ADVERSARIAL REDUNDANCY - EVALUATION PIPELINE")
    logger.info("=" * 80)
    logger.info(f"Models directory: {models_dir}")
    logger.info(f"Natural dataset: {natural_dir}")
    logger.info(f"Filtered base dir: {filtered_base_dir}")
    logger.info(f"Output directory: {output_dir}")
    logger.info(f"Batch size: {batch_size}, Workers: {workers}")
    logger.info(f"Device: {device if device else 'auto (cuda:0 or cpu)'}")
    logger.info(f"Cutoff radii: {cutoff_radii}")
    logger.info(f"Evaluate on natural test: {eval_on_natural_test}")
    logger.info(f"Evaluate on filtered test: {eval_on_filtered_test}")
    logger.info(f"Adversarial evaluation: {adv_eval}")
    logger.info("=" * 80)
    
    # Track all results
    results = {
        'config': {
            'models_dir': models_dir,
            'natural_dir': natural_dir,
            'filtered_base_dir': filtered_base_dir,
            'batch_size': batch_size,
            'workers': workers,
            'device': device,
            'cutoff_radii': cutoff_radii,
            'eval_on_natural_test': eval_on_natural_test,
            'eval_on_filtered_test': eval_on_filtered_test,
            'adv_eval': adv_eval,
        },
        'evaluations': {},
        'start_time': datetime.now().isoformat(),
    }
    
    # Create output directory
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # Load natural test data for natural test evaluation
    if eval_on_natural_test:
        logger.info("\nLoading natural test dataset...")
        natural_test_images, natural_test_labels, _ = load_natural_dataset(natural_dir, "test")
        logger.info(f"Loaded {len(natural_test_images)} natural test images")
    
    # =========================================================================
    # Evaluate on Natural Test Set
    # =========================================================================
    if eval_on_natural_test:
        logger.info("\n" + "=" * 60)
        logger.info("EVALUATING ON NATURAL TEST SET")
        logger.info("=" * 60)
        
        # Create natural dataset for evaluation
        natural_train_images, natural_train_labels, _ = load_natural_dataset(natural_dir, "train")
        natural_dataset = FilteredCIFAR10(
            data_path=natural_dir,
            dataset_type="natural",
            train_images=natural_train_images,
            train_labels=natural_train_labels,
            test_images=natural_test_images,
            test_labels=natural_test_labels,
        )
        
        # Evaluate natural model on natural test set
        if eval_natural:
            natural_model_dir = Path(models_dir) / "resnet18_natural"
            natural_checkpoint = find_checkpoint_path(natural_model_dir)
            if natural_checkpoint is not None:
                logger.info(f"\nEvaluating natural model on natural test set...")
                try:
                    eval_results = evaluate_model(
                        checkpoint_path=str(natural_checkpoint),
                        dataset=natural_dataset,
                        model_name="natural_on_natural",
                        batch_size=batch_size,
                        workers=workers,
                        device=device,
                        adv_eval=adv_eval,
                        logger=logger,
                    )
                    results['evaluations']['natural_on_natural'] = eval_results
                    logger.info(f"✓ Natural model evaluated on natural test set")
                except Exception as e:
                    logger.error(f"Failed to evaluate natural model: {e}")
                    results['evaluations']['natural_on_natural'] = {'error': str(e)}
            else:
                logger.warning(f"Natural model checkpoint not found in: {natural_model_dir} (checked for checkpoint.pt.best and checkpoint.pt)")
        
        # Evaluate filtered models on natural test set
        if eval_filtered:
            for radius in cutoff_radii:
                for variance in ['high_variance', 'low_variance']:
                    model_name = f"resnet18_{variance}_r{radius}"
                    model_dir = Path(models_dir) / model_name
                    checkpoint_path = find_checkpoint_path(model_dir)
                    
                    if checkpoint_path is not None:
                        logger.info(f"\nEvaluating {model_name} on natural test set...")
                        try:
                            eval_results = evaluate_model(
                                checkpoint_path=str(checkpoint_path),
                                dataset=natural_dataset,
                                model_name=f"{model_name}_on_natural",
                                batch_size=batch_size,
                                workers=workers,
                                device=device,
                                adv_eval=adv_eval,
                                logger=logger,
                            )
                            results['evaluations'][f"{model_name}_on_natural"] = eval_results
                            logger.info(f"✓ {model_name} evaluated on natural test set")
                        except Exception as e:
                            logger.error(f"Failed to evaluate {model_name}: {e}")
                            results['evaluations'][f"{model_name}_on_natural"] = {'error': str(e)}
                    else:
                        logger.warning(f"Model checkpoint not found in: {model_dir} (checked for checkpoint.pt.best and checkpoint.pt)")
    
    # =========================================================================
    # Evaluate on Filtered Test Sets
    # =========================================================================
    if eval_on_filtered_test:
        logger.info("\n" + "=" * 60)
        logger.info("EVALUATING ON FILTERED TEST SETS")
        logger.info("=" * 60)
        
        for radius in cutoff_radii:
            filtered_dir = Path(filtered_base_dir) / f"filtered_r{radius}"
            if not filtered_dir.exists():
                logger.warning(f"Filtered directory not found: {filtered_dir}")
                continue
            
            for variance in ['high_variance', 'low_variance']:
                # Find the test dataset directory
                test_dirs = list(filtered_dir.glob(f"cifar10_test_{variance}_r{radius}*"))
                if not test_dirs:
                    logger.warning(f"Test directory not found for {variance} r={radius}")
                    continue
                
                test_dir = test_dirs[0]
                
                # Load filtered test data
                logger.info(f"\nLoading filtered test data: {test_dir}")
                filtered_test_images, filtered_test_labels, _ = load_filtered_dataset(str(test_dir))
                
                # Find corresponding train directory for dataset creation
                train_dirs = list(filtered_dir.glob(f"cifar10_train_{variance}_r{radius}*"))
                if train_dirs:
                    train_dir = train_dirs[0]
                    filtered_train_images, filtered_train_labels, _ = load_filtered_dataset(str(train_dir))
                else:
                    # Fallback: use test data as train data (for dataset structure)
                    filtered_train_images, filtered_train_labels = filtered_test_images, filtered_test_labels
                
                # Create filtered dataset
                filtered_dataset = FilteredCIFAR10(
                    data_path=str(test_dir),
                    dataset_type=f"{variance}_r{radius}",
                    train_images=filtered_train_images,
                    train_labels=filtered_train_labels,
                    test_images=filtered_test_images,
                    test_labels=filtered_test_labels,
                )
                
                # Evaluate models trained on this filtered dataset
                model_name = f"resnet18_{variance}_r{radius}"
                model_dir = Path(models_dir) / model_name
                checkpoint_path = find_checkpoint_path(model_dir)
                
                if checkpoint_path is not None:
                    logger.info(f"\nEvaluating {model_name} on {variance} r={radius} test set...")
                    try:
                        eval_results = evaluate_model(
                            checkpoint_path=str(checkpoint_path),
                            dataset=filtered_dataset,
                            model_name=f"{model_name}_on_{variance}_r{radius}",
                            batch_size=batch_size,
                            workers=workers,
                            device=device,
                            adv_eval=adv_eval,
                            logger=logger,
                        )
                        results['evaluations'][f"{model_name}_on_{variance}_r{radius}"] = eval_results
                        logger.info(f"✓ {model_name} evaluated on {variance} r={radius} test set")
                    except Exception as e:
                        logger.error(f"Failed to evaluate {model_name}: {e}")
                        results['evaluations'][f"{model_name}_on_{variance}_r{radius}"] = {'error': str(e)}
                else:
                    logger.warning(f"Model checkpoint not found in: {model_dir} (checked for checkpoint.pt.best and checkpoint.pt)")
                
                # Evaluate natural model on this filtered test set
                if eval_natural:
                    natural_model_dir = Path(models_dir) / "resnet18_natural"
                    natural_checkpoint = find_checkpoint_path(natural_model_dir)
                    if natural_checkpoint is not None:
                        logger.info(f"\nEvaluating natural model on {variance} r={radius} test set...")
                        try:
                            eval_results = evaluate_model(
                                checkpoint_path=str(natural_checkpoint),
                                dataset=filtered_dataset,
                                model_name=f"natural_on_{variance}_r{radius}",
                                batch_size=batch_size,
                                workers=workers,
                                device=device,
                                adv_eval=adv_eval,
                                logger=logger,
                            )
                            results['evaluations'][f"natural_on_{variance}_r{radius}"] = eval_results
                            logger.info(f"✓ Natural model evaluated on {variance} r={radius} test set")
                        except Exception as e:
                            logger.error(f"Failed to evaluate natural model on {variance} r={radius}: {e}")
                            results['evaluations'][f"natural_on_{variance}_r{radius}"] = {'error': str(e)}
                    else:
                        logger.warning(f"Natural model checkpoint not found in: {natural_model_dir} (checked for checkpoint.pt.best and checkpoint.pt)")
    
    # =========================================================================
    # Save Final Results
    # =========================================================================
    results['end_time'] = datetime.now().isoformat()
    
    results_path = Path(output_dir) / "evaluation_results.json"
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    logger.info("\n" + "=" * 80)
    logger.info("EVALUATION PIPELINE COMPLETE")
    logger.info("=" * 80)
    logger.info(f"Results saved to: {results_path}")
    logger.info(f"Models evaluated: {len(results['evaluations'])}")
    for name, info in results['evaluations'].items():
        if 'error' in info:
            logger.info(f"  ✗ {name}: FAILED - {info['error']}")
        else:
            acc = info.get('natural_accuracy', 'N/A')
            logger.info(f"  ✓ {name}: {acc:.2f}% accuracy" if isinstance(acc, float) else f"  ✓ {name}: {acc}")
    
    return results


def run_evaluation_pipeline(
    models_dir: str,
    natural_dir: str,
    filtered_base_dir: str,
    output_dir: str,
    log_dir: str,
    batch_size: int = 128,
    workers: int = 4,
    device: Optional[str] = None,
    cutoff_radii: List[int] = [5, 10, 15],
    eval_natural: bool = True,
    eval_filtered: bool = True,
    eval_on_natural_test: bool = True,
    eval_on_filtered_test: bool = True,
    adv_eval: int = 0,
) -> Dict[str, Any]:
    """
    Run the complete evaluation pipeline.
    
    This is a convenience wrapper around evaluate_all_models().
    """
    return evaluate_all_models(
        models_dir=models_dir,
        natural_dir=natural_dir,
        filtered_base_dir=filtered_base_dir,
        output_dir=output_dir,
        log_dir=log_dir,
        batch_size=batch_size,
        workers=workers,
        device=device,
        cutoff_radii=cutoff_radii,
        eval_natural=eval_natural,
        eval_filtered=eval_filtered,
        eval_on_natural_test=eval_on_natural_test,
        eval_on_filtered_test=eval_on_filtered_test,
        adv_eval=adv_eval,
    )


def main():
    """
    Main entry point for the evaluation pipeline.
    """
    parser = argparse.ArgumentParser(
        description="Evaluate trained ResNet18 models on test datasets",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Required paths
    parser.add_argument(
        "--models-dir",
        type=str,
        default="./models",
        help="Directory containing trained model checkpoints"
    )
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
        default="./evaluation_results",
        help="Directory to save evaluation results"
    )
    parser.add_argument(
        "--log-dir",
        type=str,
        default="./logs/evaluation",
        help="Directory for evaluation logs"
    )
    
    # Evaluation hyperparameters
    parser.add_argument(
        "--batch-size",
        type=int,
        default=128,
        help="Evaluation batch size"
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=4,
        help="Number of data loading workers"
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device to use for evaluation (e.g., 'cuda:0', 'cuda:1', 'cpu', or '0', '1' for GPU index)"
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
        help="Skip evaluating natural model"
    )
    parser.add_argument(
        "--no-filtered",
        action="store_true",
        help="Skip evaluating filtered models"
    )
    parser.add_argument(
        "--natural-only",
        action="store_true",
        help="Only evaluate natural model"
    )
    parser.add_argument(
        "--filtered-only",
        action="store_true",
        help="Only evaluate filtered models"
    )
    
    # Test set selection
    parser.add_argument(
        "--no-natural-test",
        action="store_true",
        help="Skip evaluation on natural test set"
    )
    parser.add_argument(
        "--no-filtered-test",
        action="store_true",
        help="Skip evaluation on filtered test sets"
    )
    parser.add_argument(
        "--natural-test-only",
        action="store_true",
        help="Only evaluate on natural test set"
    )
    parser.add_argument(
        "--filtered-test-only",
        action="store_true",
        help="Only evaluate on filtered test sets"
    )
    
    # Adversarial evaluation
    parser.add_argument(
        "--adv-eval",
        action="store_true",
        help="Perform adversarial evaluation in addition to natural"
    )
    
    args = parser.parse_args()
    
    # NOTE: CUDA_VISIBLE_DEVICES must be set BEFORE PyTorch is imported to take effect.
    # The bash script (run_evaluation.sh) handles this by setting it before running Python.
    # If running Python directly, set CUDA_VISIBLE_DEVICES as an environment variable
    # before running, or use the bash script.
    #
    # After CUDA_VISIBLE_DEVICES is set, the specified GPU appears as cuda:0 to PyTorch.
    # All code should use cuda:0 after that point.
    
    # Check if CUDA_VISIBLE_DEVICES is set and log it
    if 'CUDA_VISIBLE_DEVICES' in os.environ:
        print(f"CUDA_VISIBLE_DEVICES is set to: {os.environ['CUDA_VISIBLE_DEVICES']}")
        print(f"GPU {os.environ['CUDA_VISIBLE_DEVICES']} will appear as cuda:0 to PyTorch")
    
    # Normalize device argument - if a specific GPU was requested and CUDA_VISIBLE_DEVICES
    # is set, we should use cuda:0 (the GPU will appear as cuda:0 after CUDA_VISIBLE_DEVICES)
    if args.device is not None and args.device != 'cpu':
        if args.device.isdigit() or (args.device.startswith('cuda:') and args.device != 'cuda:0'):
            # Device is a specific GPU, but after CUDA_VISIBLE_DEVICES it appears as cuda:0
            if 'CUDA_VISIBLE_DEVICES' not in os.environ:
                print("WARNING: CUDA_VISIBLE_DEVICES not set. Device specification may not work correctly.")
                print("         Use the bash script (run_evaluation.sh) or set CUDA_VISIBLE_DEVICES before running.")
            # Always use cuda:0 after CUDA_VISIBLE_DEVICES is set
            args.device = 'cuda:0'
    
    # Determine what to evaluate
    eval_natural = not args.no_natural and not args.filtered_only
    eval_filtered = not args.no_filtered and not args.natural_only
    eval_on_natural_test = not args.no_natural_test and not args.filtered_test_only
    eval_on_filtered_test = not args.no_filtered_test and not args.natural_test_only
    
    # Run the pipeline
    results = run_evaluation_pipeline(
        models_dir=args.models_dir,
        natural_dir=args.natural_dir,
        filtered_base_dir=args.filtered_base_dir,
        output_dir=args.output_dir,
        log_dir=args.log_dir,
        batch_size=args.batch_size,
        workers=args.workers,
        device=args.device,
        cutoff_radii=args.cutoff_radii,
        eval_natural=eval_natural,
        eval_filtered=eval_filtered,
        eval_on_natural_test=eval_on_natural_test,
        eval_on_filtered_test=eval_on_filtered_test,
        adv_eval=1 if args.adv_eval else 0,
    )
    
    # Exit with error code if any evaluation failed
    failed = sum(1 for info in results['evaluations'].values() if 'error' in info)
    if failed > 0:
        print(f"\n{failed} model(s) failed to evaluate. Check logs for details.")
        sys.exit(1)


if __name__ == "__main__":
    main()

