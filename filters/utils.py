"""
Utility functions for loading and managing CIFAR-10 dataset.

This module provides utilities for:
- Loading CIFAR-10 from torchvision
- Saving CIFAR-10 to disk in various formats
- Creating custom dataloaders
- Dataset statistics and visualization
"""

import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, transforms
from pathlib import Path
from typing import Tuple, Optional, List
import pickle


class CIFAR10Dataset(Dataset):
    """
    Custom CIFAR-10 dataset wrapper for PyTorch.
    
    This class provides a flexible interface for CIFAR-10 data,
    supporting both torchvision datasets and numpy arrays.
    """
    
    def __init__(
        self,
        images: np.ndarray,
        labels: np.ndarray,
        transform: Optional[transforms.Compose] = None
    ):
        """
        Initialize CIFAR-10 dataset.
        
        Args:
            images: Numpy array of images, shape (N, H, W, C) or (N, C, H, W)
            labels: Numpy array of labels, shape (N,)
            transform: Optional torchvision transforms to apply
        """
        self.images = images
        self.labels = labels
        self.transform = transform
        
        # Ensure images are in (N, H, W, C) format
        if len(self.images.shape) == 4 and self.images.shape[1] == 3:
            # Convert from (N, C, H, W) to (N, H, W, C)
            self.images = np.transpose(self.images, (0, 2, 3, 1))
    
    def __len__(self) -> int:
        return len(self.images)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        image = self.images[idx]
        label = self.labels[idx]
        
        # Apply transforms if provided
        if self.transform:
            image = self.transform(image)
        else:
            # Convert to tensor and normalize to [0, 1]
            if image.dtype == np.uint8:
                image = image.astype(np.float32) / 255.0
            image = torch.from_numpy(image).permute(2, 0, 1)  # (H, W, C) -> (C, H, W)
        
        return image, label


def load_cifar10(
    root: str = "./data",
    train: bool = True,
    download: bool = True,
    transform: Optional[transforms.Compose] = None
) -> datasets.CIFAR10:
    """
    Load CIFAR-10 dataset using torchvision.
    
    CIFAR-10 consists of 60,000 32x32 color images in 10 classes:
    - airplane, automobile, bird, cat, deer, dog, frog, horse, ship, truck
    - 50,000 training images
    - 10,000 test images
    
    Args:
        root: Directory to save/load dataset
        train: If True, load training set; otherwise load test set
        download: If True, download dataset if not already present
        transform: Optional torchvision transforms
        
    Returns:
        datasets.CIFAR10: PyTorch CIFAR-10 dataset
    """
    if transform is None:
        # Default: just convert to tensor (no normalization)
        transform = transforms.Compose([
            transforms.ToTensor(),
        ])
    
    dataset = datasets.CIFAR10(
        root=root,
        train=train,
        download=download,
        transform=transform
    )
    
    return dataset


def save_cifar10(
    dataset: datasets.CIFAR10,
    output_dir: str,
    dataset_name: str = "cifar10"
) -> str:
    """
    Save CIFAR-10 dataset to disk in numpy format.
    
    This function extracts all images and labels from a PyTorch dataset
    and saves them as numpy arrays for faster loading and processing.
    
    Args:
        dataset: PyTorch CIFAR-10 dataset
        output_dir: Directory to save the dataset
        dataset_name: Name prefix for saved files
        
    Returns:
        str: Path to output directory
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    print(f"Saving CIFAR-10 dataset to {output_path}...")
    
    # Extract all images and labels
    images = []
    labels = []
    
    for idx in range(len(dataset)):
        image, label = dataset[idx]
        
        # Convert tensor to numpy
        if torch.is_tensor(image):
            # Assume image is in [C, H, W] format
            image = image.permute(1, 2, 0).numpy()  # Convert to (H, W, C)
        
        # Ensure uint8 format
        if image.max() <= 1.0:
            image = (image * 255).astype(np.uint8)
        else:
            image = image.astype(np.uint8)
        
        images.append(image)
        labels.append(label)
    
    # Convert to numpy arrays
    images = np.stack(images)  # Shape: (N, H, W, C)
    labels = np.array(labels)  # Shape: (N,)
    
    # Save to disk
    images_path = output_path / f"{dataset_name}_images.npy"
    labels_path = output_path / f"{dataset_name}_labels.npy"
    
    np.save(images_path, images)
    np.save(labels_path, labels)
    
    # Save metadata
    metadata = {
        'num_images': len(images),
        'image_shape': images[0].shape,
        'num_classes': len(np.unique(labels)),
        'class_names': ['airplane', 'automobile', 'bird', 'cat', 'deer',
                       'dog', 'frog', 'horse', 'ship', 'truck']
    }
    
    metadata_path = output_path / f"{dataset_name}_metadata.npy"
    np.save(metadata_path, metadata)
    
    print(f"✓ Saved {len(images)} images to {images_path}")
    print(f"✓ Saved {len(labels)} labels to {labels_path}")
    print(f"✓ Saved metadata to {metadata_path}")
    print(f"  - Image shape: {images.shape}")
    print(f"  - Labels shape: {labels.shape}")
    
    return str(output_path)


def load_saved_cifar10(
    data_dir: str,
    dataset_name: str = "cifar10"
) -> Tuple[np.ndarray, np.ndarray, dict]:
    """
    Load CIFAR-10 dataset from saved numpy files.
    
    Args:
        data_dir: Directory containing saved dataset
        dataset_name: Name prefix used when saving
        
    Returns:
        Tuple[np.ndarray, np.ndarray, dict]: (images, labels, metadata)
    """
    data_path = Path(data_dir)
    
    images = np.load(data_path / f"{dataset_name}_images.npy")
    labels = np.load(data_path / f"{dataset_name}_labels.npy")
    metadata = np.load(data_path / f"{dataset_name}_metadata.npy", allow_pickle=True).item()
    
    return images, labels, metadata


def create_dataloader(
    dataset: Dataset,
    batch_size: int = 128,
    shuffle: bool = True,
    num_workers: int = 4
) -> DataLoader:
    """
    Create a PyTorch DataLoader for a dataset.
    
    Args:
        dataset: PyTorch dataset
        batch_size: Number of samples per batch
        shuffle: Whether to shuffle data
        num_workers: Number of worker processes for data loading
        
    Returns:
        DataLoader: PyTorch DataLoader
    """
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True
    )


def get_cifar10_statistics(dataset: Dataset) -> dict:
    """
    Compute statistics (mean, std) for CIFAR-10 dataset.
    
    These statistics are useful for normalization during training.
    
    Args:
        dataset: CIFAR-10 dataset
        
    Returns:
        dict: Dictionary containing mean and std per channel
    """
    print("Computing dataset statistics...")
    
    # Collect all images
    images = []
    for idx in range(len(dataset)):
        image, _ = dataset[idx]
        if torch.is_tensor(image):
            image = image.numpy()
        else:
            if image.dtype == np.uint8:
                image = image.astype(np.float32) / 255.0
            # Assume (H, W, C) format, convert to (C, H, W)
            if image.shape[-1] == 3:
                image = np.transpose(image, (2, 0, 1))
        images.append(image)
    
    images = np.stack(images)  # Shape: (N, C, H, W)
    
    # Compute mean and std per channel
    mean = np.mean(images, axis=(0, 2, 3))
    std = np.std(images, axis=(0, 2, 3))
    
    statistics = {
        'mean': mean.tolist(),
        'std': std.tolist(),
        'mean_rgb': f"[{mean[0]:.4f}, {mean[1]:.4f}, {mean[2]:.4f}]",
        'std_rgb': f"[{std[0]:.4f}, {std[1]:.4f}, {std[2]:.4f}]"
    }
    
    print(f"✓ Mean: {statistics['mean_rgb']}")
    print(f"✓ Std:  {statistics['std_rgb']}")
    
    return statistics


def get_standard_transforms(
    augment: bool = False,
    normalize: bool = True
) -> transforms.Compose:
    """
    Get standard image transforms for CIFAR-10.
    
    Args:
        augment: If True, include data augmentation (for training)
        normalize: If True, normalize with CIFAR-10 statistics
        
    Returns:
        transforms.Compose: Composed transforms
    """
    # CIFAR-10 standard statistics
    mean = [0.4914, 0.4822, 0.4465]
    std = [0.2470, 0.2435, 0.2616]
    
    transform_list = []
    
    if augment:
        transform_list.extend([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
        ])
    
    transform_list.append(transforms.ToTensor())
    
    if normalize:
        transform_list.append(transforms.Normalize(mean, std))
    
    return transforms.Compose(transform_list)


if __name__ == "__main__":
    """
    Example usage and testing.
    """
    print("="*60)
    print("CIFAR-10 Data Utilities - Example Usage")
    print("="*60)
    
    # 1. Load CIFAR-10 train and test sets
    print("\n1. Loading CIFAR-10 datasets...")
    train_dataset = load_cifar10(root="./data", train=True, download=True)
    test_dataset = load_cifar10(root="./data", train=False, download=True)
    
    print(f"   Train size: {len(train_dataset)}")
    print(f"   Test size: {len(test_dataset)}")
    
    # 2. Save datasets to disk
    print("\n2. Saving datasets to disk...")
    train_dir = save_cifar10(train_dataset, "./saved_data", "cifar10_train")
    test_dir = save_cifar10(test_dataset, "./saved_data", "cifar10_test")
    
    # 3. Load saved datasets
    print("\n3. Loading saved datasets...")
    train_images, train_labels, train_metadata = load_saved_cifar10("./saved_data", "cifar10_train")
    print(f"   Loaded train images: {train_images.shape}")
    print(f"   Loaded train labels: {train_labels.shape}")
    print(f"   Metadata: {train_metadata}")
    
    # 4. Compute statistics
    print("\n4. Computing dataset statistics...")
    stats = get_cifar10_statistics(train_dataset)
    
    # 5. Create dataloader
    print("\n5. Creating dataloader...")
    train_loader = create_dataloader(train_dataset, batch_size=128, shuffle=True)
    print(f"   Number of batches: {len(train_loader)}")
    
    # Test one batch
    images, labels = next(iter(train_loader))
    print(f"   Batch images shape: {images.shape}")
    print(f"   Batch labels shape: {labels.shape}")
    
    print("\n" + "="*60)
    print("Utility functions tested successfully!")
    print("="*60)

