"""
Direct Variance Test

This script performs a direct pixel-wise variance comparison to verify that:
1. The low-pass (high-variance) dataset captures most of the original variance (~90%)
2. The high-pass (low-variance) dataset captures minimal variance (~10%)

This is the most straightforward validation method and provides clear, quantitative results.
"""

import numpy as np
import argparse
from pathlib import Path
import sys
from typing import Tuple, Dict

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))
from filters.dft import load_filtered_dataset
from filters.utils import load_cifar10


def calculate_pixelwise_variance(images: np.ndarray) -> float:
    """
    Calculate total pixel-wise variance across all images.
    
    Args:
        images: Array of images, shape (N, H, W, C)
        
    Returns:
        float: Total variance across all pixels
    """
    # Ensure images are float
    if images.dtype == np.uint8:
        images = images.astype(np.float32) / 255.0
    elif images.max() > 1.0:
        images = images.astype(np.float32) / 255.0
    
    # Calculate variance across the batch dimension for each pixel
    # Result shape: (H, W, C)
    pixel_variances = np.var(images, axis=0)
    
    # Sum all pixel variances to get total variance
    total_variance = np.sum(pixel_variances)
    
    return total_variance


def run_variance_test(
    natural_dataset_dir: str,
    filtered_dataset_dir: str,
    cutoff_radius: int,
    n_samples: int = 1000,
    output_dir: str = "./validation_results"
) -> Dict[str, float]:
    """
    Run the direct variance test on datasets.
    
    Args:
        natural_dataset_dir: Path to original CIFAR-10 dataset
        filtered_dataset_dir: Path to filtered datasets directory
        cutoff_radius: Cutoff radius used for filtering
        n_samples: Number of samples to use for variance calculation
        output_dir: Directory to save results
        
    Returns:
        Dict with variance results
    """
    print("="*70)
    print("DIRECT VARIANCE TEST")
    print("="*70)
    print(f"\nConfiguration:")
    print(f"  - Natural dataset: {natural_dataset_dir}")
    print(f"  - Filtered dataset: {filtered_dataset_dir}")
    print(f"  - Cutoff radius: {cutoff_radius}")
    print(f"  - Number of samples: {n_samples}")
    print()
    
    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Step 1: Load original CIFAR-10 dataset
    print("Step 1: Loading original CIFAR-10 dataset...")
    try:
        train_dataset = load_cifar10(root=natural_dataset_dir, train=True, download=False)
    except:
        print("  Error: Could not load natural dataset from torchvision.")
        print("  Attempting to load from saved numpy files...")
        from filters.utils import load_saved_cifar10
        images_orig, labels_orig, _ = load_saved_cifar10(natural_dataset_dir, "cifar10_train")
        images_orig = images_orig[:n_samples]
    else:
        # Extract images from dataset
        images_orig = []
        for idx in range(min(n_samples, len(train_dataset))):
            img, _ = train_dataset[idx]
            if hasattr(img, 'permute'):  # torch tensor
                img = img.permute(1, 2, 0).numpy()
            if img.max() <= 1.0:
                img = (img * 255).astype(np.uint8)
            images_orig.append(img)
        images_orig = np.array(images_orig)
    
    print(f"  ✓ Loaded {len(images_orig)} original images")
    print(f"  ✓ Image shape: {images_orig[0].shape}")
    
    # Step 2: Calculate original variance
    print("\nStep 2: Calculating original dataset variance...")
    variance_original = calculate_pixelwise_variance(images_orig)
    print(f"  ✓ Original variance: {variance_original:.2f}")
    
    # Step 3: Load and calculate high-variance (low-pass) dataset variance
    print("\nStep 3: Loading high-variance (low-pass filtered) dataset...")
    
    # Try both naming conventions
    radius_str = str(cutoff_radius)
    high_var_paths = [
        Path(filtered_dataset_dir) / f"cifar10_train_high_variance_r{radius_str}",
        Path(filtered_dataset_dir) / f"cifar10_train_high_variance_r{radius_str}.0"
    ]
    
    high_var_dir = None
    for path in high_var_paths:
        if path.exists():
            high_var_dir = str(path)
            break
    
    if high_var_dir is None:
        print(f"  ✗ Error: High-variance dataset not found")
        print(f"    Tried: {[str(p) for p in high_var_paths]}")
        return None
    
    images_high_var, _, metadata = load_filtered_dataset(high_var_dir)
    images_high_var = images_high_var[:n_samples]
    print(f"  ✓ Loaded {len(images_high_var)} high-variance images")
    
    variance_high = calculate_pixelwise_variance(images_high_var)
    print(f"  ✓ High-variance dataset variance: {variance_high:.2f}")
    
    # Step 4: Load and calculate low-variance (high-pass) dataset variance
    print("\nStep 4: Loading low-variance (high-pass filtered) dataset...")
    
    low_var_paths = [
        Path(filtered_dataset_dir) / f"cifar10_train_low_variance_r{radius_str}",
        Path(filtered_dataset_dir) / f"cifar10_train_low_variance_r{radius_str}.0"
    ]
    
    low_var_dir = None
    for path in low_var_paths:
        if path.exists():
            low_var_dir = str(path)
            break
    
    if low_var_dir is None:
        print(f"  ✗ Error: Low-variance dataset not found")
        print(f"    Tried: {[str(p) for p in low_var_paths]}")
        return None
    
    images_low_var, _, metadata = load_filtered_dataset(low_var_dir)
    images_low_var = images_low_var[:n_samples]
    print(f"  ✓ Loaded {len(images_low_var)} low-variance images")
    
    variance_low = calculate_pixelwise_variance(images_low_var)
    print(f"  ✓ Low-variance dataset variance: {variance_low:.2f}")
    
    # Step 5: Calculate percentages and display results
    print("\n" + "="*70)
    print("RESULTS")
    print("="*70)
    
    pct_high = (variance_high / variance_original) * 100
    pct_low = (variance_low / variance_original) * 100
    pct_total = pct_high + pct_low
    
    results = {
        'variance_original': variance_original,
        'variance_high': variance_high,
        'variance_low': variance_low,
        'percentage_high': pct_high,
        'percentage_low': pct_low,
        'percentage_total': pct_total,
        'cutoff_radius': cutoff_radius,
        'n_samples': n_samples
    }
    
    print(f"\nVariance Analysis (cutoff radius r={cutoff_radius}):")
    print(f"  Original dataset variance:      {variance_original:12.2f} (100.00%)")
    print(f"  High-variance dataset variance: {variance_high:12.2f} ({pct_high:6.2f}%)")
    print(f"  Low-variance dataset variance:  {variance_low:12.2f} ({pct_low:6.2f}%)")
    print(f"  Total accounted:                              ({pct_total:6.2f}%)")
    
    print(f"\nInterpretation:")
    if pct_high > 80:
        print(f"  ✓ PASS: High-variance dataset captures {pct_high:.1f}% of variance")
        print(f"          This confirms the low-pass filter isolates high-variance components.")
    else:
        print(f"  ⚠ WARNING: High-variance dataset only captures {pct_high:.1f}% of variance")
        print(f"             Expected >80%. Consider adjusting cutoff radius.")
    
    if pct_low < 20:
        print(f"  ✓ PASS: Low-variance dataset captures only {pct_low:.1f}% of variance")
        print(f"          This confirms the high-pass filter isolates low-variance components.")
    else:
        print(f"  ⚠ WARNING: Low-variance dataset captures {pct_low:.1f}% of variance")
        print(f"             Expected <20%. Consider adjusting cutoff radius.")
    
    # Step 6: Save results to file
    results_file = output_path / f"variance_test_r{cutoff_radius}.txt"
    print(f"\nSaving results to: {results_file}")
    
    with open(results_file, 'w') as f:
        f.write("="*70 + "\n")
        f.write("DIRECT VARIANCE TEST RESULTS\n")
        f.write("="*70 + "\n\n")
        f.write(f"Configuration:\n")
        f.write(f"  Cutoff radius: {cutoff_radius}\n")
        f.write(f"  Number of samples: {n_samples}\n")
        f.write(f"  Natural dataset: {natural_dataset_dir}\n")
        f.write(f"  Filtered dataset: {filtered_dataset_dir}\n\n")
        f.write("Variance Analysis:\n")
        f.write(f"  Original dataset variance:      {variance_original:12.2f} (100.00%)\n")
        f.write(f"  High-variance dataset variance: {variance_high:12.2f} ({pct_high:6.2f}%)\n")
        f.write(f"  Low-variance dataset variance:  {variance_low:12.2f} ({pct_low:6.2f}%)\n")
        f.write(f"  Total accounted:                              ({pct_total:6.2f}%)\n\n")
        f.write("Interpretation:\n")
        if pct_high > 80:
            f.write(f"  ✓ PASS: High-variance dataset captures {pct_high:.1f}% of variance\n")
        else:
            f.write(f"  ⚠ WARNING: High-variance dataset only captures {pct_high:.1f}%\n")
        if pct_low < 20:
            f.write(f"  ✓ PASS: Low-variance dataset captures only {pct_low:.1f}% of variance\n")
        else:
            f.write(f"  ⚠ WARNING: Low-variance dataset captures {pct_low:.1f}%\n")
    
    print(f"  ✓ Results saved")
    print()
    
    return results


def main():
    parser = argparse.ArgumentParser(
        description="Direct Variance Test for filtered datasets",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument(
        "--natural-dataset-dir",
        type=str,
        default="./data/cifar10_natural",
        help="Path to original CIFAR-10 dataset"
    )
    
    parser.add_argument(
        "--filtered-dataset-dir",
        type=str,
        default="./data/filtered_r10",
        help="Path to filtered datasets directory"
    )
    
    parser.add_argument(
        "--cutoff-radius",
        type=int,
        default=10,
        help="Cutoff radius used for filtering"
    )
    
    parser.add_argument(
        "--n-samples",
        type=int,
        default=1000,
        help="Number of samples to use (for computational efficiency)"
    )
    
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./validation_results",
        help="Directory to save results"
    )
    
    args = parser.parse_args()
    
    results = run_variance_test(
        natural_dataset_dir=args.natural_dataset_dir,
        filtered_dataset_dir=args.filtered_dataset_dir,
        cutoff_radius=args.cutoff_radius,
        n_samples=args.n_samples,
        output_dir=args.output_dir
    )
    
    if results:
        print("="*70)
        print("VALIDATION COMPLETE")
        print("="*70)
    else:
        print("\n✗ Validation failed - check error messages above")
        sys.exit(1)


if __name__ == "__main__":
    main()

