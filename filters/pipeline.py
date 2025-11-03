"""
Complete pipeline for downloading CIFAR-10 and creating filtered datasets.

This script orchestrates the entire workflow:
1. Downloads CIFAR-10 to a specified directory
2. Creates high-variance (low-pass) and low-variance (high-pass) filtered datasets
3. Saves filtered datasets to a specified output directory
4. Processes both training and test sets
"""

import argparse
import os
from pathlib import Path
from typing import Optional

from utils import load_cifar10, save_cifar10
from dft import process_dataset


def run_pipeline(
    natural_dataset_dir: str = "./data/cifar10_natural",
    filtered_dataset_dir: str = "./data/cifar10_filtered",
    cutoff_radius: float = 10.0,
    use_butterworth: bool = True,
    butterworth_order: int = 2,
    download: bool = True
) -> None:
    """
    Run the complete DFT filtering pipeline.
    
    Args:
        natural_dataset_dir: Directory to save/load the original CIFAR-10 dataset
        filtered_dataset_dir: Directory to save the filtered datasets
        cutoff_radius: Radial frequency cutoff for filtering
        use_butterworth: Whether to use Butterworth filter (recommended)
        butterworth_order: Order of Butterworth filter
        download: Whether to download CIFAR-10 if not present
    """
    print("="*70)
    print("DFT FILTERING PIPELINE")
    print("="*70)
    print(f"\nConfiguration:")
    print(f"  - Natural dataset directory: {natural_dataset_dir}")
    print(f"  - Filtered dataset directory: {filtered_dataset_dir}")
    print(f"  - Cutoff radius: {cutoff_radius}")
    print(f"  - Filter type: {'Butterworth' if use_butterworth else 'Circular'}")
    if use_butterworth:
        print(f"  - Butterworth order: {butterworth_order}")
    print()
    
    # Create output directories
    Path(natural_dataset_dir).mkdir(parents=True, exist_ok=True)
    Path(filtered_dataset_dir).mkdir(parents=True, exist_ok=True)
    
    # Step 1: Load CIFAR-10 training dataset
    print("-" * 70)
    print("STEP 1: Loading CIFAR-10 Training Dataset")
    print("-" * 70)
    train_dataset = load_cifar10(
        root=natural_dataset_dir,
        train=True,
        download=download
    )
    print(f"✓ Loaded {len(train_dataset)} training images\n")
    
    # Step 2: Load CIFAR-10 test dataset
    print("-" * 70)
    print("STEP 2: Loading CIFAR-10 Test Dataset")
    print("-" * 70)
    test_dataset = load_cifar10(
        root=natural_dataset_dir,
        train=False,
        download=download
    )
    print(f"✓ Loaded {len(test_dataset)} test images\n")
    
    # Step 3: Save natural datasets (optional, for faster reloading)
    print("-" * 70)
    print("STEP 3: Saving Natural Datasets")
    print("-" * 70)
    save_cifar10(train_dataset, natural_dataset_dir, "cifar10_train")
    print()
    save_cifar10(test_dataset, natural_dataset_dir, "cifar10_test")
    print()
    
    # Step 4: Process training dataset with DFT filtering
    print("-" * 70)
    print("STEP 4: Processing Training Dataset with DFT Filtering")
    print("-" * 70)
    train_high_var_dir, train_low_var_dir = process_dataset(
        dataset=train_dataset,
        output_dir=filtered_dataset_dir,
        cutoff_radius=cutoff_radius,
        use_butterworth=use_butterworth,
        butterworth_order=butterworth_order,
        dataset_name="cifar10_train"
    )
    print(f"\n✓ Training datasets created:")
    print(f"  - High-variance: {train_high_var_dir}")
    print(f"  - Low-variance:  {train_low_var_dir}\n")
    
    # Step 5: Process test dataset with DFT filtering
    print("-" * 70)
    print("STEP 5: Processing Test Dataset with DFT Filtering")
    print("-" * 70)
    test_high_var_dir, test_low_var_dir = process_dataset(
        dataset=test_dataset,
        output_dir=filtered_dataset_dir,
        cutoff_radius=cutoff_radius,
        use_butterworth=use_butterworth,
        butterworth_order=butterworth_order,
        dataset_name="cifar10_test"
    )
    print(f"\n✓ Test datasets created:")
    print(f"  - High-variance: {test_high_var_dir}")
    print(f"  - Low-variance:  {test_low_var_dir}\n")
    
    # Summary
    print("="*70)
    print("PIPELINE COMPLETE")
    print("="*70)
    print(f"\nSummary:")
    print(f"  ✓ Downloaded/loaded CIFAR-10 dataset to: {natural_dataset_dir}")
    print(f"  ✓ Created filtered datasets in: {filtered_dataset_dir}")
    print(f"  ✓ Processed {len(train_dataset)} training images")
    print(f"  ✓ Processed {len(test_dataset)} test images")
    print(f"\nDatasets created:")
    print(f"  1. Training - High-variance: {train_high_var_dir}")
    print(f"  2. Training - Low-variance:  {train_low_var_dir}")
    print(f"  3. Test - High-variance:     {test_high_var_dir}")
    print(f"  4. Test - Low-variance:      {test_low_var_dir}")
    print(f"\nNext steps:")
    print(f"  - Train models on high-variance and low-variance datasets")
    print(f"  - Create ensemble from trained models")
    print(f"  - Test adversarial robustness")
    print()


def main():
    """
    Main entry point with command-line argument parsing.
    """
    parser = argparse.ArgumentParser(
        description="DFT Filtering Pipeline for CIFAR-10",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Required arguments
    parser.add_argument(
        "--natural-dataset-dir",
        type=str,
        default="./data/cifar10_natural",
        help="Directory to save/load the original CIFAR-10 dataset"
    )
    
    parser.add_argument(
        "--filtered-dataset-dir",
        type=str,
        default="./data/cifar10_filtered",
        help="Directory to save the filtered datasets"
    )
    
    # Filtering parameters
    parser.add_argument(
        "--cutoff-radius",
        type=float,
        default=10.0,
        help="Radial frequency cutoff for filtering (recommended: 5-15)"
    )
    
    parser.add_argument(
        "--use-butterworth",
        action="store_true",
        default=True,
        help="Use Butterworth filter (smooth transition, reduces ringing)"
    )
    
    parser.add_argument(
        "--no-butterworth",
        action="store_false",
        dest="use_butterworth",
        help="Use circular mask instead of Butterworth filter"
    )
    
    parser.add_argument(
        "--butterworth-order",
        type=int,
        default=2,
        help="Order of Butterworth filter (higher = sharper transition)"
    )
    
    parser.add_argument(
        "--no-download",
        action="store_false",
        dest="download",
        help="Don't download CIFAR-10 if not present (will fail if not found)"
    )
    
    args = parser.parse_args()
    
    # Run the pipeline
    run_pipeline(
        natural_dataset_dir=args.natural_dataset_dir,
        filtered_dataset_dir=args.filtered_dataset_dir,
        cutoff_radius=args.cutoff_radius,
        use_butterworth=args.use_butterworth,
        butterworth_order=args.butterworth_order,
        download=args.download
    )


if __name__ == "__main__":
    """
    Example usage:
    
    # Basic usage with default settings:
    python pipeline.py
    
    # Custom directories:
    python pipeline.py --natural-dataset-dir ./my_data/natural --filtered-dataset-dir ./my_data/filtered
    
    # Different cutoff radius:
    python pipeline.py --cutoff-radius 15
    
    # Use circular mask instead of Butterworth:
    python pipeline.py --no-butterworth
    
    # Multiple experiments with different radii:
    for r in 5 10 15; do
        python pipeline.py --cutoff-radius $r --filtered-dataset-dir ./data/filtered_r${r}
    done
    """
    main()

