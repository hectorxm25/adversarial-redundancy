"""
2D Discrete Fourier Transform (DFT) filtering module for CIFAR-10 images.

This module implements 2D DFT-based filtering to separate high-variance (low-frequency)
and low-variance (high-frequency) components of images. The implementation follows
the procedure outlined in the referenced paper, which shows that under translational
invariance assumptions, the Fourier basis is equivalent to the PCA basis.

Key Features:
- Per-channel (R, G, B) 2D FFT processing
- Butterworth filter for smooth frequency transitions
- Both low-pass (high-variance) and high-pass (low-variance) filtering
- Batch processing of entire datasets
"""

import os
import numpy as np
import torch
from typing import Tuple, Optional, Literal
from pathlib import Path
from tqdm import tqdm


def create_butterworth_mask(
    shape: Tuple[int, int],
    cutoff_radius: float,
    order: int = 2,
    high_pass: bool = False
) -> np.ndarray:
    """
    Create a Butterworth filter mask to avoid ringing effects from hard cutoffs.
    
    The Butterworth filter provides a smooth transition between pass and stop bands,
    which reduces ringing artifacts compared to a hard circular mask.
    
    Args:
        shape: Shape of the mask (height, width)
        cutoff_radius: Radial frequency cutoff (distance from center)
        order: Order of the Butterworth filter (higher = sharper transition)
        high_pass: If True, create high-pass filter; otherwise low-pass
        
    Returns:
        np.ndarray: Butterworth filter mask of shape (height, width)
    """
    rows, cols = shape
    center_row, center_col = rows // 2, cols // 2
    
    # Create coordinate grids
    y, x = np.ogrid[:rows, :cols]
    
    # Calculate distance from center for each point
    distance = np.sqrt((y - center_row)**2 + (x - center_col)**2)
    
    # Butterworth filter formula
    if high_pass:
        # High-pass: H(u,v) = 1 / (1 + (D0/D(u,v))^(2n))
        # Avoid division by zero
        epsilon = 1e-8
        mask = 1.0 / (1.0 + (cutoff_radius / (distance + epsilon))**(2 * order))
    else:
        # Low-pass: H(u,v) = 1 / (1 + (D(u,v)/D0)^(2n))
        mask = 1.0 / (1.0 + (distance / cutoff_radius)**(2 * order))
    
    return mask


def create_circular_mask(
    shape: Tuple[int, int],
    cutoff_radius: float,
    high_pass: bool = False
) -> np.ndarray:
    """
    Create a hard circular mask (box filter).
    
    Note: This can create ringing effects. Consider using Butterworth filter instead.
    
    Args:
        shape: Shape of the mask (height, width)
        cutoff_radius: Radial frequency cutoff (distance from center)
        high_pass: If True, create high-pass filter; otherwise low-pass
        
    Returns:
        np.ndarray: Binary mask of shape (height, width)
    """
    rows, cols = shape
    center_row, center_col = rows // 2, cols // 2
    
    # Create coordinate grids
    y, x = np.ogrid[:rows, :cols]
    
    # Calculate distance from center for each point
    distance = np.sqrt((y - center_row)**2 + (x - center_col)**2)
    
    if high_pass:
        mask = (distance > cutoff_radius).astype(np.float32)
    else:
        mask = (distance <= cutoff_radius).astype(np.float32)
    
    return mask


def apply_dft_filter_to_image(
    image: np.ndarray,
    cutoff_radius: float,
    filter_type: Literal['low_pass', 'high_pass'] = 'low_pass',
    use_butterworth: bool = True,
    butterworth_order: int = 2
) -> np.ndarray:
    """
    Apply 2D DFT filtering to a single image.
    
    This function processes each color channel (R, G, B) independently:
    1. Apply 2D FFT to each channel
    2. Shift zero-frequency to center
    3. Apply frequency mask (low-pass or high-pass)
    4. Inverse shift and inverse FFT
    5. Take real part and recombine channels
    
    Args:
        image: Input image of shape (H, W, 3) with values in [0, 255] or [0, 1]
        cutoff_radius: Radial frequency cutoff
        filter_type: 'low_pass' for high-variance or 'high_pass' for low-variance
        use_butterworth: If True, use Butterworth filter; otherwise use hard circular mask
        butterworth_order: Order of Butterworth filter (only used if use_butterworth=True)
        
    Returns:
        np.ndarray: Filtered image of same shape as input
    """
    # Ensure image is float for processing
    if image.dtype == np.uint8:
        image = image.astype(np.float32) / 255.0
        was_uint8 = True
    else:
        image = image.astype(np.float32)
        was_uint8 = False
    
    h, w, c = image.shape
    filtered_image = np.zeros_like(image)
    
    # Create the frequency mask
    high_pass = (filter_type == 'high_pass')
    if use_butterworth:
        mask = create_butterworth_mask((h, w), cutoff_radius, butterworth_order, high_pass)
    else:
        mask = create_circular_mask((h, w), cutoff_radius, high_pass)
    
    # Process each channel independently
    for channel_idx in range(c):
        channel = image[:, :, channel_idx]
        
        # Step 1: Apply 2D FFT
        fft_channel = np.fft.fft2(channel)
        
        # Step 2: Shift zero-frequency to center
        fft_shifted = np.fft.fftshift(fft_channel)
        
        # Step 3: Apply mask
        filtered_fft_shifted = fft_shifted * mask
        
        # Step 4: Inverse shift
        filtered_fft = np.fft.ifftshift(filtered_fft_shifted)
        
        # Step 5: Inverse FFT
        filtered_channel = np.fft.ifft2(filtered_fft)
        
        # Step 6: Take real part
        filtered_image[:, :, channel_idx] = np.real(filtered_channel)
    
    # Clip values to valid range
    filtered_image = np.clip(filtered_image, 0, 1)
    
    # Convert back to uint8 if input was uint8
    if was_uint8:
        filtered_image = (filtered_image * 255).astype(np.uint8)
    
    return filtered_image


def process_dataset(
    dataset: torch.utils.data.Dataset,
    output_dir: str,
    cutoff_radius: float,
    use_butterworth: bool = True,
    butterworth_order: int = 2,
    dataset_name: str = "cifar10"
) -> Tuple[str, str]:
    """
    Process an entire dataset and create both high-variance and low-variance versions.
    
    This function:
    1. Creates output directories for high-variance and low-variance datasets
    2. Processes each image in the dataset
    3. Applies low-pass filter to create high-variance dataset
    4. Applies high-pass filter to create low-variance dataset
    5. Saves both versions with labels
    
    Args:
        dataset: PyTorch dataset (e.g., CIFAR-10) with images and labels
        output_dir: Root directory to save filtered datasets
        cutoff_radius: Radial frequency cutoff for filtering
        use_butterworth: Whether to use Butterworth filter (recommended)
        butterworth_order: Order of Butterworth filter
        dataset_name: Name of the dataset (for directory naming)
        
    Returns:
        Tuple[str, str]: Paths to (high_variance_dir, low_variance_dir)
    """
    # Create output directories
    output_path = Path(output_dir)
    high_var_dir = output_path / f"{dataset_name}_high_variance_r{cutoff_radius}"
    low_var_dir = output_path / f"{dataset_name}_low_variance_r{cutoff_radius}"
    
    high_var_dir.mkdir(parents=True, exist_ok=True)
    low_var_dir.mkdir(parents=True, exist_ok=True)
    
    # Store images and labels
    high_var_images = []
    low_var_images = []
    labels = []
    
    print(f"Processing {len(dataset)} images with cutoff radius r={cutoff_radius}...")
    print(f"Using {'Butterworth' if use_butterworth else 'Circular'} filter")
    
    # Process each image
    for idx in tqdm(range(len(dataset)), desc="Filtering images"):
        # Get image and label
        image, label = dataset[idx]
        
        # Convert tensor to numpy if needed
        if torch.is_tensor(image):
            # Assume image is in [C, H, W] format (PyTorch standard)
            image = image.permute(1, 2, 0).numpy()
        
        # Ensure image is in correct range
        if image.max() <= 1.0:
            image = (image * 255).astype(np.uint8)
        else:
            image = image.astype(np.uint8)
        
        # Apply filters
        high_var_image = apply_dft_filter_to_image(
            image, cutoff_radius, 'low_pass', use_butterworth, butterworth_order
        )
        low_var_image = apply_dft_filter_to_image(
            image, cutoff_radius, 'high_pass', use_butterworth, butterworth_order
        )
        
        # Store results
        high_var_images.append(high_var_image)
        low_var_images.append(low_var_image)
        labels.append(label)
    
    # Convert to numpy arrays
    high_var_images = np.stack(high_var_images)
    low_var_images = np.stack(low_var_images)
    labels = np.array(labels)
    
    # Save datasets
    print(f"Saving high-variance dataset to {high_var_dir}...")
    np.save(high_var_dir / "images.npy", high_var_images)
    np.save(high_var_dir / "labels.npy", labels)
    
    print(f"Saving low-variance dataset to {low_var_dir}...")
    np.save(low_var_dir / "images.npy", low_var_images)
    np.save(low_var_dir / "labels.npy", labels)
    
    # Save metadata
    metadata = {
        'cutoff_radius': cutoff_radius,
        'use_butterworth': use_butterworth,
        'butterworth_order': butterworth_order if use_butterworth else None,
        'num_images': len(dataset),
        'image_shape': high_var_images[0].shape,
    }
    
    np.save(high_var_dir / "metadata.npy", metadata)
    np.save(low_var_dir / "metadata.npy", metadata)
    
    print(f"✓ High-variance dataset saved: {high_var_dir}")
    print(f"✓ Low-variance dataset saved: {low_var_dir}")
    print(f"  - Images shape: {high_var_images.shape}")
    print(f"  - Labels shape: {labels.shape}")
    
    return str(high_var_dir), str(low_var_dir)


def load_filtered_dataset(dataset_dir: str) -> Tuple[np.ndarray, np.ndarray, dict]:
    """
    Load a previously filtered dataset.
    
    Args:
        dataset_dir: Directory containing the filtered dataset
        
    Returns:
        Tuple[np.ndarray, np.ndarray, dict]: (images, labels, metadata)
    """
    dataset_path = Path(dataset_dir)
    
    images = np.load(dataset_path / "images.npy")
    labels = np.load(dataset_path / "labels.npy")
    metadata = np.load(dataset_path / "metadata.npy", allow_pickle=True).item()
    
    return images, labels, metadata


if __name__ == "__main__":
    """
    Example usage and testing.
    """
    import sys
    
    # Try to import utils for testing
    try:
        from utils import load_cifar10, save_cifar10
    except ImportError:
        print("utils.py not found. Please create it first to test the full pipeline.")
        sys.exit(1)
    
    print("="*60)
    print("DFT Filtering Pipeline - Example Usage")
    print("="*60)
    
    # Load CIFAR-10
    print("\n1. Loading CIFAR-10 dataset...")
    train_dataset, test_dataset = load_cifar10(download=True)
    print(f"   Train size: {len(train_dataset)}, Test size: {len(test_dataset)}")
    
    # Process with different cutoff radii (hyperparameter exploration)
    cutoff_radii = [5, 10, 15]
    
    for r in cutoff_radii:
        print(f"\n2. Processing dataset with cutoff radius r={r}...")
        high_var_dir, low_var_dir = process_dataset(
            train_dataset,
            output_dir="./filtered_data",
            cutoff_radius=r,
            use_butterworth=True,
            butterworth_order=2,
            dataset_name="cifar10_train"
        )
        
        print(f"\n3. Verifying saved data...")
        high_var_images, high_var_labels, metadata = load_filtered_dataset(high_var_dir)
        print(f"   Successfully loaded high-variance dataset: {high_var_images.shape}")
        
    print("\n" + "="*60)
    print("Pipeline execution complete!")
    print("="*60)

