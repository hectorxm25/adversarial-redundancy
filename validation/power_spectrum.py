"""
Power Spectrum Plot

This script creates a Fourier power spectrum plot that recreates Figure A.12 from the paper.
It shows the sharp "elbow" and "long tail" that justifies the filtering approach.

The power spectrum demonstrates that variance is concentrated in low-frequency
components (the "head") with minimal variance in high-frequency components (the "tail").
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import argparse
from pathlib import Path
import sys
from typing import Tuple

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))
from filters.utils import load_cifar10


def compute_power_spectrum(images: np.ndarray) -> np.ndarray:
    """
    Compute the 2D power spectrum for a batch of images.
    
    Args:
        images: Array of images, shape (N, H, W, C)
        
    Returns:
        np.ndarray: Average power spectrum, shape (H, W, C)
    """
    # Ensure images are float
    if images.dtype == np.uint8:
        images = images.astype(np.float32) / 255.0
    elif images.max() > 1.0:
        images = images.astype(np.float32) / 255.0
    
    n_images = len(images)
    h, w, c = images[0].shape
    
    # Store power spectrums for each image
    power_spectrums = []
    
    print(f"  Computing power spectrum for {n_images} images...")
    
    for idx in range(n_images):
        if (idx + 1) % 10 == 0:
            print(f"    Processed {idx + 1}/{n_images} images...")
        
        image = images[idx]
        image_power = np.zeros((h, w, c))
        
        # Process each channel separately
        for channel_idx in range(c):
            channel = image[:, :, channel_idx]
            
            # Apply 2D FFT
            fft_channel = np.fft.fft2(channel)
            
            # Shift zero-frequency to center
            fft_shifted = np.fft.fftshift(fft_channel)
            
            # Compute power (squared magnitude)
            power = np.abs(fft_shifted) ** 2
            
            image_power[:, :, channel_idx] = power
        
        power_spectrums.append(image_power)
    
    # Average across all images
    avg_power_spectrum = np.mean(power_spectrums, axis=0)
    
    return avg_power_spectrum


def create_power_spectrum_plot(
    natural_dataset_dir: str,
    n_samples: int = 50,
    output_dir: str = "./validation_results",
    cutoff_radii: list = None
) -> None:
    """
    Create a power spectrum plot from the original dataset.
    
    Args:
        natural_dataset_dir: Path to original CIFAR-10 dataset
        n_samples: Number of images to use for spectrum computation
        output_dir: Directory to save the plot
        cutoff_radii: Optional list of cutoff radii to mark on the plot
    """
    print("="*70)
    print("POWER SPECTRUM PLOT")
    print("="*70)
    print(f"\nConfiguration:")
    print(f"  - Natural dataset: {natural_dataset_dir}")
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
        images, labels, _ = load_saved_cifar10(natural_dataset_dir, "cifar10_train")
        images = images[:n_samples]
    else:
        # Extract images from dataset
        images = []
        for idx in range(min(n_samples, len(train_dataset))):
            img, _ = train_dataset[idx]
            if hasattr(img, 'permute'):  # torch tensor
                img = img.permute(1, 2, 0).numpy()
            if img.max() <= 1.0:
                img = (img * 255).astype(np.uint8)
            images.append(img)
        images = np.array(images)
    
    print(f"  ✓ Loaded {len(images)} images")
    print(f"  ✓ Image shape: {images[0].shape}")
    
    # Step 2: Compute power spectrum
    print("\nStep 2: Computing 2D power spectrum...")
    avg_power_spectrum = compute_power_spectrum(images)
    print(f"  ✓ Power spectrum computed")
    print(f"  ✓ Spectrum shape: {avg_power_spectrum.shape}")
    
    # Step 3: Sort power values
    print("\nStep 3: Sorting power spectrum values...")
    # Average across channels for overall power
    power_per_freq = np.mean(avg_power_spectrum, axis=2)
    
    # Flatten and sort in descending order
    power_sorted = np.sort(power_per_freq.flatten())[::-1]
    
    print(f"  ✓ Total frequency components: {len(power_sorted)}")
    print(f"  ✓ Max power: {power_sorted[0]:.2e}")
    print(f"  ✓ Min power: {power_sorted[-1]:.2e}")
    print(f"  ✓ Power range (log10): {np.log10(power_sorted[0]):.2f} to {np.log10(power_sorted[-1]):.2f}")
    
    # Step 4: Create the plot
    print("\nStep 4: Creating power spectrum plot...")
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Plot sorted power values on log scale
    indices = np.arange(len(power_sorted))
    ax.plot(indices, power_sorted, linewidth=2, color='steelblue', label='Power spectrum')
    
    # Mark cutoff radii if provided
    if cutoff_radii:
        # For a 32x32 image, the maximum radius from center is ~22.6
        # Map radius to approximate component index
        max_radius = np.sqrt(2) * 16  # diagonal from center
        colors = ['red', 'orange', 'green']
        for i, r in enumerate(cutoff_radii):
            # Rough approximation: components scale with area of circle
            approx_idx = int((r / max_radius)**2 * len(power_sorted))
            color = colors[i % len(colors)]
            ax.axvline(approx_idx, color=color, linestyle='--', linewidth=1.5,
                      label=f'Cutoff r={r} (~{approx_idx} components)', alpha=0.7)
    
    # Set y-axis to log scale
    ax.set_yscale('log')
    
    # Labels and title
    ax.set_xlabel('Frequency Component Index (sorted by power)', fontsize=12)
    ax.set_ylabel('Power (log scale)', fontsize=12)
    ax.set_title(f'Power Spectrum: CIFAR-10 Frequency Components\n'
                 f'({n_samples} images averaged)', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3, which='both')
    ax.legend(fontsize=10)
    
    # Add annotation about the "elbow" and "tail"
    # Find approximate elbow point (where power drops significantly)
    elbow_idx = np.where(power_sorted < power_sorted[0] * 0.01)[0]
    if len(elbow_idx) > 0:
        elbow_idx = elbow_idx[0]
        ax.annotate('Sharp "elbow"\n(head components)', 
                   xy=(elbow_idx, power_sorted[elbow_idx]),
                   xytext=(elbow_idx + 100, power_sorted[elbow_idx] * 10),
                   arrowprops=dict(arrowstyle='->', color='black', lw=1.5),
                   fontsize=10, ha='left')
        
        tail_idx = len(power_sorted) - 100
        ax.annotate('Long "tail"\n(low-variance components)', 
                   xy=(tail_idx, power_sorted[tail_idx]),
                   xytext=(tail_idx - 200, power_sorted[tail_idx] * 100),
                   arrowprops=dict(arrowstyle='->', color='black', lw=1.5),
                   fontsize=10, ha='right')
    
    plt.tight_layout()
    
    # Save plot
    output_file = output_path / "power_spectrum_plot.png"
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"  ✓ Plot saved to: {output_file}")
    
    # Step 5: Save numerical results
    print("\nStep 5: Saving numerical results...")
    results_file = output_path / "power_spectrum_data.txt"
    
    with open(results_file, 'w') as f:
        f.write("="*70 + "\n")
        f.write("POWER SPECTRUM ANALYSIS RESULTS\n")
        f.write("="*70 + "\n\n")
        f.write(f"Configuration:\n")
        f.write(f"  Number of samples: {n_samples}\n")
        f.write(f"  Natural dataset: {natural_dataset_dir}\n")
        f.write(f"  Image size: {images[0].shape}\n\n")
        f.write("Power Spectrum Statistics:\n")
        f.write(f"  Total frequency components: {len(power_sorted)}\n")
        f.write(f"  Maximum power: {power_sorted[0]:.6e}\n")
        f.write(f"  Minimum power: {power_sorted[-1]:.6e}\n")
        f.write(f"  Power ratio (max/min): {power_sorted[0]/power_sorted[-1]:.2e}\n\n")
        
        # Calculate cumulative variance
        cumulative_power = np.cumsum(power_sorted)
        total_power = cumulative_power[-1]
        
        f.write("Cumulative Power Distribution:\n")
        for pct in [50, 80, 90, 95, 99]:
            threshold = total_power * (pct / 100)
            n_components = np.searchsorted(cumulative_power, threshold) + 1
            f.write(f"  {pct}% of power in first {n_components:4d} components ")
            f.write(f"({n_components/len(power_sorted)*100:.1f}% of all components)\n")
        
        f.write("\nInterpretation:\n")
        f.write("  The sharp 'elbow' in the power spectrum confirms that variance\n")
        f.write("  is concentrated in low-frequency components. This justifies the\n")
        f.write("  filtering approach: low-pass filters capture high-variance\n")
        f.write("  components, while high-pass filters capture low-variance components.\n")
        
        if cutoff_radii:
            f.write(f"\nCutoff Radii Analysis:\n")
            for r in cutoff_radii:
                approx_idx = int((r / max_radius)**2 * len(power_sorted))
                power_captured = cumulative_power[min(approx_idx, len(cumulative_power)-1)]
                pct_captured = (power_captured / total_power) * 100
                f.write(f"  r={r}: Captures ~{pct_captured:.1f}% of total power\n")
    
    print(f"  ✓ Results saved to: {results_file}")
    
    # Print interpretation
    print("\n" + "="*70)
    print("INTERPRETATION")
    print("="*70)
    print("\nThis power spectrum plot recreates Figure A.12 from the paper.")
    print("It demonstrates:")
    print("  1. Sharp 'elbow': Most variance is in low-frequency components")
    print("  2. Long 'tail': High-frequency components have minimal variance")
    print("  3. This justifies splitting data into high- and low-variance datasets")
    print()
    
    # Calculate and display cumulative stats
    cumulative_power = np.cumsum(power_sorted)
    total_power = cumulative_power[-1]
    
    print("Key Statistics:")
    for pct in [50, 80, 90, 95]:
        threshold = total_power * (pct / 100)
        n_components = np.searchsorted(cumulative_power, threshold) + 1
        print(f"  {pct}% of power concentrated in first {n_components} of {len(power_sorted)} components")
        print(f"    ({n_components/len(power_sorted)*100:.1f}% of all frequency components)")
    
    print()


def main():
    parser = argparse.ArgumentParser(
        description="Power Spectrum Plot for Fourier analysis validation",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument(
        "--natural-dataset-dir",
        type=str,
        default="./data/cifar10_natural",
        help="Path to original CIFAR-10 dataset"
    )
    
    parser.add_argument(
        "--n-samples",
        type=int,
        default=50,
        help="Number of samples to use (50 is sufficient per paper)"
    )
    
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./validation_results",
        help="Directory to save results and plots"
    )
    
    parser.add_argument(
        "--cutoff-radii",
        type=int,
        nargs='+',
        default=[5, 10, 15],
        help="Cutoff radii to mark on the plot (optional)"
    )
    
    args = parser.parse_args()
    
    try:
        create_power_spectrum_plot(
            natural_dataset_dir=args.natural_dataset_dir,
            n_samples=args.n_samples,
            output_dir=args.output_dir,
            cutoff_radii=args.cutoff_radii
        )
        
        print("="*70)
        print("VALIDATION COMPLETE")
        print("="*70)
    except Exception as e:
        print(f"\n✗ Validation failed with error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()

