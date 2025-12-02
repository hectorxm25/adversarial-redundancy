#!/usr/bin/env python3
"""
Inference script for the Redundancy Ensemble.

This script runs inference on a dataset using the ensemble of natural,
high-variance, and low-variance models.

Usage:
    # Evaluate on natural CIFAR-10 test set with weak voting
    python inference.py --dataset natural --radius 10 --voting weak
    
    # Evaluate on filtered test set with strong voting
    python inference.py --dataset high_variance --radius 15 --voting strong
    
    # Evaluate on adversarial examples
    python inference.py --dataset adversarial --adversarial-type pgd --radius 10
"""

import os
import sys
import gc
import argparse
import json
import logging
from pathlib import Path
from datetime import datetime
from typing import Optional, Dict, Any, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from ensemble.ensemble import RedundancyEnsemble, NoClearWinner, load_ensemble
from train.datasets import NumpyDataset, load_natural_dataset, load_filtered_dataset


def setup_logging(log_dir: str, run_name: str) -> logging.Logger:
    """
    Set up logging for inference.
    
    Args:
        log_dir: Directory for log files
        run_name: Name of the inference run
        
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
    """Clean up GPU memory."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()


def get_test_transform() -> transforms.Compose:
    """Get the standard test transform for CIFAR-10."""
    return transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.4914, 0.4822, 0.4465],
            std=[0.2470, 0.2435, 0.2616]
        ),
    ])


def load_dataset(
    dataset_type: str,
    natural_dir: str,
    filtered_base_dir: str,
    adversarial_dir: Optional[str] = None,
    cutoff_radius: int = 10,
    split: str = "test",
    adversarial_type: str = "pgd",
    adversarial_model: str = "natural",
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load a dataset for inference.
    
    Args:
        dataset_type: One of 'natural', 'high_variance', 'low_variance', 'adversarial'
        natural_dir: Path to natural CIFAR-10 dataset
        filtered_base_dir: Base path for filtered datasets
        adversarial_dir: Path to adversarial examples (required if dataset_type='adversarial')
        cutoff_radius: Cutoff radius for filtered datasets
        split: 'train' or 'test'
        adversarial_type: Type of adversarial examples ('pgd' or 'fgsm')
        adversarial_model: Which model's adversarial examples to use
        
    Returns:
        Tuple of (images, labels) as numpy arrays
    """
    if dataset_type == 'natural':
        images, labels, _ = load_natural_dataset(natural_dir, split)
        
    elif dataset_type in ['high_variance', 'low_variance']:
        filtered_dir = Path(filtered_base_dir) / f"filtered_r{cutoff_radius}"
        
        # Find the actual directory (may have .0 suffix)
        search_pattern = f"cifar10_{split}_{dataset_type}_r{cutoff_radius}*"
        matching_dirs = list(filtered_dir.glob(search_pattern))
        
        if not matching_dirs:
            raise FileNotFoundError(
                f"Could not find {dataset_type} dataset in {filtered_dir} "
                f"matching pattern: {search_pattern}"
            )
        
        dataset_dir = matching_dirs[0]
        images, labels, _ = load_filtered_dataset(str(dataset_dir))
        
    elif dataset_type == 'adversarial':
        if adversarial_dir is None:
            raise ValueError("adversarial_dir is required when dataset_type='adversarial'")
        
        # Construct the path to adversarial examples
        adv_path = Path(adversarial_dir) / f"resnet18_{adversarial_model}" / adversarial_type / split
        
        if not adv_path.exists():
            raise FileNotFoundError(f"Adversarial examples not found at: {adv_path}")
        
        images = np.load(adv_path / "adversarial_images.npy")
        labels = np.load(adv_path / "labels.npy")
        
    else:
        raise ValueError(f"Unknown dataset type: {dataset_type}")
    
    return images, labels


def run_inference(
    ensemble: RedundancyEnsemble,
    images: np.ndarray,
    labels: np.ndarray,
    batch_size: int = 128,
    workers: int = 4,
    logger: Optional[logging.Logger] = None,
) -> Dict[str, Any]:
    """
    Run inference on a dataset and compute metrics.
    
    Args:
        ensemble: The RedundancyEnsemble model
        images: Input images, shape (N, H, W, C)
        labels: True labels, shape (N,)
        batch_size: Batch size for inference
        workers: Number of data loading workers
        logger: Optional logger
        
    Returns:
        Dictionary containing inference results
    """
    if logger is None:
        logger = logging.getLogger("inference")
    
    logger.info(f"Running inference on {len(images)} images...")
    
    # Create dataset and dataloader
    transform = get_test_transform()
    dataset = NumpyDataset(images, labels, transform=transform)
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=workers,
        pin_memory=True,
    )
    
    # Run inference
    all_voted_preds = []
    all_natural_preds = []
    all_high_var_preds = []
    all_low_var_preds = []
    all_labels = []
    no_winner_count = 0
    no_winner_indices = []
    
    device = ensemble._device
    
    with torch.no_grad():
        for batch_idx, (batch_images, batch_labels) in enumerate(tqdm(dataloader, desc="Running inference")):
            batch_images = batch_images.to(device)
            batch_labels = batch_labels.to(device)
            
            try:
                voted_preds, natural_preds, high_var_preds, low_var_preds = ensemble(
                    batch_images, return_all_predictions=True
                )
                
                all_voted_preds.append(voted_preds.cpu())
                all_natural_preds.append(natural_preds.cpu())
                all_high_var_preds.append(high_var_preds.cpu())
                all_low_var_preds.append(low_var_preds.cpu())
                all_labels.append(batch_labels.cpu())
                
            except NoClearWinner as e:
                # In strong mode, this will be raised for each image without unanimous vote
                # We need to handle batch-level processing
                logger.warning(f"NoClearWinner in batch {batch_idx}: {e.message}")
                
                # For batches with NoClearWinner, process individually
                for i in range(batch_images.shape[0]):
                    single_image = batch_images[i:i+1]
                    single_label = batch_labels[i:i+1]
                    
                    try:
                        voted_pred, nat_pred, high_pred, low_pred = ensemble(
                            single_image, return_all_predictions=True
                        )
                        all_voted_preds.append(voted_pred.cpu())
                        all_natural_preds.append(nat_pred.cpu())
                        all_high_var_preds.append(high_pred.cpu())
                        all_low_var_preds.append(low_pred.cpu())
                        all_labels.append(single_label.cpu())
                    except NoClearWinner:
                        no_winner_count += 1
                        no_winner_indices.append(batch_idx * batch_size + i)
                        # For no winner cases, we'll mark with -1
                        all_voted_preds.append(torch.tensor([-1]))
                        
                        # Still get individual predictions for analysis
                        nat_logits, high_logits, low_logits = ensemble._get_predictions(single_image)
                        all_natural_preds.append(nat_logits.argmax(dim=1).cpu())
                        all_high_var_preds.append(high_logits.argmax(dim=1).cpu())
                        all_low_var_preds.append(low_logits.argmax(dim=1).cpu())
                        all_labels.append(single_label.cpu())
    
    # Concatenate results
    all_voted_preds = torch.cat(all_voted_preds)
    all_natural_preds = torch.cat(all_natural_preds)
    all_high_var_preds = torch.cat(all_high_var_preds)
    all_low_var_preds = torch.cat(all_low_var_preds)
    all_labels = torch.cat(all_labels)
    
    # Compute metrics
    valid_mask = all_voted_preds != -1  # Exclude NoClearWinner cases
    
    # Ensemble accuracy (only on cases where we got a prediction)
    if valid_mask.sum() > 0:
        ensemble_correct = (all_voted_preds[valid_mask] == all_labels[valid_mask]).sum().item()
        ensemble_acc = ensemble_correct / valid_mask.sum().item() * 100
    else:
        ensemble_acc = 0.0
        ensemble_correct = 0
    
    # Individual model accuracies
    natural_acc = (all_natural_preds == all_labels).float().mean().item() * 100
    high_var_acc = (all_high_var_preds == all_labels).float().mean().item() * 100
    low_var_acc = (all_low_var_preds == all_labels).float().mean().item() * 100
    
    # Agreement statistics
    nat_high_agree = (all_natural_preds == all_high_var_preds).float().mean().item() * 100
    nat_low_agree = (all_natural_preds == all_low_var_preds).float().mean().item() * 100
    high_low_agree = (all_high_var_preds == all_low_var_preds).float().mean().item() * 100
    all_agree = ((all_natural_preds == all_high_var_preds) & 
                 (all_high_var_preds == all_low_var_preds)).float().mean().item() * 100
    
    results = {
        'total_samples': len(images),
        'valid_predictions': valid_mask.sum().item(),
        'no_winner_count': no_winner_count,
        'no_winner_rate': no_winner_count / len(images) * 100,
        'accuracies': {
            'ensemble': ensemble_acc,
            'natural': natural_acc,
            'high_variance': high_var_acc,
            'low_variance': low_var_acc,
        },
        'agreement': {
            'natural_high_variance': nat_high_agree,
            'natural_low_variance': nat_low_agree,
            'high_low_variance': high_low_agree,
            'all_three': all_agree,
        },
        'predictions': {
            'voted': all_voted_preds.numpy().tolist(),
            'natural': all_natural_preds.numpy().tolist(),
            'high_variance': all_high_var_preds.numpy().tolist(),
            'low_variance': all_low_var_preds.numpy().tolist(),
            'labels': all_labels.numpy().tolist(),
        },
        'no_winner_indices': no_winner_indices,
    }
    
    # Log summary
    logger.info("=" * 60)
    logger.info("INFERENCE RESULTS")
    logger.info("=" * 60)
    logger.info(f"Total samples: {results['total_samples']}")
    logger.info(f"Valid predictions: {results['valid_predictions']}")
    logger.info(f"No clear winner: {results['no_winner_count']} ({results['no_winner_rate']:.2f}%)")
    logger.info("-" * 60)
    logger.info("Accuracies:")
    logger.info(f"  Ensemble: {results['accuracies']['ensemble']:.2f}%")
    logger.info(f"  Natural: {results['accuracies']['natural']:.2f}%")
    logger.info(f"  High-variance: {results['accuracies']['high_variance']:.2f}%")
    logger.info(f"  Low-variance: {results['accuracies']['low_variance']:.2f}%")
    logger.info("-" * 60)
    logger.info("Model Agreement:")
    logger.info(f"  Natural & High-variance: {results['agreement']['natural_high_variance']:.2f}%")
    logger.info(f"  Natural & Low-variance: {results['agreement']['natural_low_variance']:.2f}%")
    logger.info(f"  High-variance & Low-variance: {results['agreement']['high_low_variance']:.2f}%")
    logger.info(f"  All three agree: {results['agreement']['all_three']:.2f}%")
    logger.info("=" * 60)
    
    return results


def main():
    """Main entry point for the inference script."""
    parser = argparse.ArgumentParser(
        description="Run inference using the Redundancy Ensemble",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Required arguments
    parser.add_argument(
        "--radius", "-r",
        type=int,
        required=True,
        choices=[5, 10, 15],
        help="Cutoff radius for filtering (5, 10, or 15)"
    )
    parser.add_argument(
        "--voting", "-v",
        type=str,
        required=True,
        choices=['weak', 'strong'],
        help="Voting mode: 'weak' (majority) or 'strong' (unanimous)"
    )
    
    # Dataset selection
    parser.add_argument(
        "--dataset", "-d",
        type=str,
        default="natural",
        choices=['natural', 'high_variance', 'low_variance', 'adversarial'],
        help="Dataset to evaluate on"
    )
    parser.add_argument(
        "--split",
        type=str,
        default="test",
        choices=['train', 'test'],
        help="Dataset split to use"
    )
    parser.add_argument(
        "--adversarial-type",
        type=str,
        default="pgd",
        choices=['pgd', 'fgsm'],
        help="Type of adversarial examples (only used if dataset='adversarial')"
    )
    parser.add_argument(
        "--adversarial-model",
        type=str,
        default="natural",
        help="Which model's adversarial examples to use (e.g., 'natural', 'high_variance_r10')"
    )
    
    # Paths
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
        help="Base path for filtered datasets"
    )
    parser.add_argument(
        "--adversarial-dir",
        type=str,
        default="./adversarial_data",
        help="Path to adversarial examples directory"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./ensemble_results",
        help="Directory to save inference results"
    )
    parser.add_argument(
        "--log-dir",
        type=str,
        default="./logs/ensemble",
        help="Directory for inference logs"
    )
    
    # Inference parameters
    parser.add_argument(
        "--batch-size",
        type=int,
        default=128,
        help="Batch size for inference"
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
        help="Device to use (e.g., 'cuda:0', 'cpu'). Default: auto-select"
    )
    
    # Output options
    parser.add_argument(
        "--save-predictions",
        action="store_true",
        help="Save individual predictions to output file"
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Only output essential information"
    )
    
    args = parser.parse_args()
    
    # Set up logging
    logger = setup_logging(args.log_dir, f"inference_r{args.radius}_{args.voting}")
    
    if not args.quiet:
        logger.info("=" * 80)
        logger.info("REDUNDANCY ENSEMBLE - INFERENCE")
        logger.info("=" * 80)
        logger.info(f"Cutoff radius: {args.radius}")
        logger.info(f"Voting mode: {args.voting}")
        logger.info(f"Dataset: {args.dataset}")
        logger.info(f"Split: {args.split}")
        logger.info(f"Device: {args.device if args.device else 'auto'}")
        logger.info("=" * 80)
    
    # Load ensemble
    logger.info("Loading ensemble...")
    try:
        ensemble = load_ensemble(
            models_dir=args.models_dir,
            cutoff_radius=args.radius,
            voting_mode=args.voting,
            device=args.device,
        )
    except FileNotFoundError as e:
        logger.error(f"Failed to load ensemble: {e}")
        sys.exit(1)
    
    # Load dataset
    logger.info(f"Loading {args.dataset} dataset ({args.split} split)...")
    try:
        images, labels = load_dataset(
            dataset_type=args.dataset,
            natural_dir=args.natural_dir,
            filtered_base_dir=args.filtered_base_dir,
            adversarial_dir=args.adversarial_dir,
            cutoff_radius=args.radius,
            split=args.split,
            adversarial_type=args.adversarial_type,
            adversarial_model=args.adversarial_model,
        )
        logger.info(f"Loaded {len(images)} images")
    except FileNotFoundError as e:
        logger.error(f"Failed to load dataset: {e}")
        sys.exit(1)
    
    # Run inference
    results = run_inference(
        ensemble=ensemble,
        images=images,
        labels=labels,
        batch_size=args.batch_size,
        workers=args.workers,
        logger=logger,
    )
    
    # Add configuration to results
    results['config'] = {
        'cutoff_radius': args.radius,
        'voting_mode': args.voting,
        'dataset': args.dataset,
        'split': args.split,
        'adversarial_type': args.adversarial_type if args.dataset == 'adversarial' else None,
        'adversarial_model': args.adversarial_model if args.dataset == 'adversarial' else None,
        'batch_size': args.batch_size,
        'timestamp': datetime.now().isoformat(),
    }
    
    # Remove predictions from output if not requested (to save space)
    if not args.save_predictions:
        del results['predictions']
        del results['no_winner_indices']
    
    # Save results
    output_path = Path(args.output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Create filename based on configuration
    if args.dataset == 'adversarial':
        filename = f"ensemble_r{args.radius}_{args.voting}_{args.dataset}_{args.adversarial_type}_{args.adversarial_model}_{args.split}.json"
    else:
        filename = f"ensemble_r{args.radius}_{args.voting}_{args.dataset}_{args.split}.json"
    
    results_file = output_path / filename
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    logger.info(f"Results saved to: {results_file}")
    
    # Print summary for quick reference
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Ensemble Accuracy: {results['accuracies']['ensemble']:.2f}%")
    print(f"Natural Model: {results['accuracies']['natural']:.2f}%")
    print(f"High-Variance Model: {results['accuracies']['high_variance']:.2f}%")
    print(f"Low-Variance Model: {results['accuracies']['low_variance']:.2f}%")
    if results['no_winner_count'] > 0:
        print(f"No Clear Winner: {results['no_winner_count']} ({results['no_winner_rate']:.2f}%)")
    print("=" * 60)
    
    # Cleanup
    del ensemble
    cleanup_gpu_memory()
    
    return results


if __name__ == "__main__":
    main()

