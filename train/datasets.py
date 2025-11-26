"""
Custom dataset classes for use with the robustness library.

This module provides dataset wrappers that are compatible with the robustness
library's training pipeline while supporting our custom numpy-based filtered datasets.
"""

import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from pathlib import Path
from typing import Tuple, Optional, Dict, Any
import sys

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from robustness.datasets import DataSet
from robustness import data_augmentation as da
from robustness import cifar_models


class NumpyDataset(Dataset):
    """
    PyTorch Dataset wrapper for numpy arrays.
    
    This class wraps numpy image and label arrays into a PyTorch Dataset
    that can be used with DataLoader.
    
    Attributes:
        images: Numpy array of images, shape (N, H, W, C)
        labels: Numpy array of labels, shape (N,)
        transform: Optional torchvision transforms to apply
    """
    
    def __init__(
        self,
        images: np.ndarray,
        labels: np.ndarray,
        transform: Optional[transforms.Compose] = None
    ):
        """
        Initialize the dataset.
        
        Args:
            images: Numpy array of images, shape (N, H, W, C) or (N, C, H, W)
            labels: Numpy array of labels, shape (N,)
            transform: Optional torchvision transforms to apply
        """
        self.images = images
        self.labels = labels
        self.transform = transform
        
        # Ensure images are in (N, H, W, C) format for PIL compatibility
        if len(self.images.shape) == 4 and self.images.shape[1] == 3:
            # Convert from (N, C, H, W) to (N, H, W, C)
            self.images = np.transpose(self.images, (0, 2, 3, 1))
    
    def __len__(self) -> int:
        return len(self.images)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        image = self.images[idx]
        label = int(self.labels[idx])
        
        # Apply transforms if provided
        if self.transform:
            # Convert to PIL Image for transforms
            from PIL import Image
            if image.dtype != np.uint8:
                if image.max() <= 1.0:
                    image = (image * 255).astype(np.uint8)
                else:
                    image = image.astype(np.uint8)
            pil_image = Image.fromarray(image)
            image = self.transform(pil_image)
        else:
            # Default: convert to tensor and normalize to [0, 1]
            if image.dtype == np.uint8:
                image = image.astype(np.float32) / 255.0
            image = torch.from_numpy(image).permute(2, 0, 1)  # (H, W, C) -> (C, H, W)
        
        return image, label


class FilteredCIFAR10(DataSet):
    """
    Custom CIFAR-10 dataset class compatible with the robustness library.
    
    This class extends the robustness library's DataSet class to support
    loading from our custom numpy-based filtered datasets while maintaining
    compatibility with the robustness training pipeline.
    
    The robustness library expects datasets to have:
    - make_loaders() method that returns train and validation dataloaders
    - Proper normalization statistics (mean, std)
    - Number of classes
    """
    
    # CIFAR-10 standard statistics
    CIFAR10_MEAN = [0.4914, 0.4822, 0.4465]
    CIFAR10_STD = [0.2470, 0.2435, 0.2616]
    
    # Class names
    CLASS_NAMES = ['airplane', 'automobile', 'bird', 'cat', 'deer',
                   'dog', 'frog', 'horse', 'ship', 'truck']
    
    def __init__(
        self,
        data_path: str,
        dataset_type: str = "natural",
        train_images: Optional[np.ndarray] = None,
        train_labels: Optional[np.ndarray] = None,
        test_images: Optional[np.ndarray] = None,
        test_labels: Optional[np.ndarray] = None,
        **kwargs
    ):
        """
        Initialize the FilteredCIFAR10 dataset.
        
        Args:
            data_path: Path to the dataset directory
            dataset_type: Type of dataset - "natural", "high_variance", or "low_variance"
            train_images: Optional pre-loaded training images
            train_labels: Optional pre-loaded training labels
            test_images: Optional pre-loaded test images
            test_labels: Optional pre-loaded test labels
            **kwargs: Additional arguments passed to parent class
        """
        self.dataset_type = dataset_type
        
        # Store pre-loaded data if provided
        self._train_images = train_images
        self._train_labels = train_labels
        self._test_images = test_images
        self._test_labels = test_labels
        
        # Store data_path as instance variable (parent class also stores it)
        self._data_path = Path(data_path)
        
        # Initialize parent class with CIFAR-10 properties
        ds_kwargs = {
            'num_classes': 10,
            'mean': torch.tensor(self.CIFAR10_MEAN),
            'std': torch.tensor(self.CIFAR10_STD),
            'custom_class': None,
            'label_mapping': None,
            'transform_train': da.TRAIN_TRANSFORMS_DEFAULT(32),
            'transform_test': da.TEST_TRANSFORMS_DEFAULT(32),
        }
        ds_kwargs.update(kwargs)
        
        # Parent class requires (ds_name, data_path, **kwargs)
        super().__init__(f'cifar10_{dataset_type}', str(data_path), **ds_kwargs)
    
    def get_model(self, arch: str, pretrained: bool = False):
        """
        Get a model architecture for this dataset.
        
        Args:
            arch: Architecture name (e.g., 'resnet18', 'resnet50')
            pretrained: Whether to use pretrained weights (not supported for CIFAR)
            
        Returns:
            A model instance with the correct number of output classes
        """
        if pretrained:
            raise ValueError('FilteredCIFAR10 does not support pytorch_pretrained=True')
        return cifar_models.__dict__[arch](num_classes=self.num_classes)
    
    def load_natural_data(self, split: str = "train") -> Tuple[np.ndarray, np.ndarray]:
        """
        Load natural CIFAR-10 data from numpy files.
        
        Args:
            split: "train" or "test"
            
        Returns:
            Tuple of (images, labels) numpy arrays
        """
        prefix = f"cifar10_{split}"
        data_path = Path(self.data_path)
        images = np.load(data_path / f"{prefix}_images.npy")
        labels = np.load(data_path / f"{prefix}_labels.npy")
        return images, labels
    
    def load_filtered_data(
        self,
        filtered_dir: str,
        split: str = "train"
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Load filtered dataset from numpy files.
        
        Args:
            filtered_dir: Path to filtered dataset directory
            split: "train" or "test"
            
        Returns:
            Tuple of (images, labels) numpy arrays
        """
        filtered_path = Path(filtered_dir)
        images = np.load(filtered_path / "images.npy")
        labels = np.load(filtered_path / "labels.npy")
        return images, labels
    
    def get_transforms(self, augment: bool = True) -> transforms.Compose:
        """
        Get image transforms for training or testing.
        
        Args:
            augment: Whether to apply data augmentation
            
        Returns:
            Composed transforms
        """
        transform_list = []
        
        if augment:
            transform_list.extend([
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
            ])
        
        transform_list.extend([
            transforms.ToTensor(),
            transforms.Normalize(self.CIFAR10_MEAN, self.CIFAR10_STD),
        ])
        
        return transforms.Compose(transform_list)
    
    def make_loaders(
        self,
        workers: int = 4,
        batch_size: int = 128,
        data_aug: bool = True,
        subset: Optional[int] = None,
        subset_seed: int = 0,
        subset_type: str = 'rand',
        val_batch_size: Optional[int] = None,
        shuffle_train: bool = True,
        shuffle_val: bool = False,
        **kwargs
    ) -> Tuple[DataLoader, DataLoader]:
        """
        Create train and validation dataloaders.
        
        This method is required by the robustness library's training pipeline.
        
        Args:
            workers: Number of data loading workers
            batch_size: Training batch size
            data_aug: Whether to apply data augmentation
            subset: Number of samples to use (None for all)
            subset_seed: Random seed for subset selection
            subset_type: Type of subset selection ('rand' or 'first')
            val_batch_size: Validation batch size (defaults to batch_size)
            shuffle_train: Whether to shuffle training data
            shuffle_val: Whether to shuffle validation data
            **kwargs: Additional arguments (ignored)
            
        Returns:
            Tuple of (train_loader, val_loader)
        """
        if val_batch_size is None:
            val_batch_size = batch_size
        
        # Load data
        if self._train_images is not None and self._train_labels is not None:
            train_images, train_labels = self._train_images, self._train_labels
        else:
            train_images, train_labels = self.load_natural_data("train")
        
        if self._test_images is not None and self._test_labels is not None:
            test_images, test_labels = self._test_images, self._test_labels
        else:
            test_images, test_labels = self.load_natural_data("test")
        
        # Apply subset if specified
        if subset is not None:
            if subset_type == 'rand':
                rng = np.random.RandomState(subset_seed)
                indices = rng.permutation(len(train_images))[:subset]
            else:
                indices = np.arange(subset)
            train_images = train_images[indices]
            train_labels = train_labels[indices]
        
        # Create datasets with transforms
        train_transform = self.get_transforms(augment=data_aug)
        test_transform = self.get_transforms(augment=False)
        
        train_dataset = NumpyDataset(train_images, train_labels, transform=train_transform)
        test_dataset = NumpyDataset(test_images, test_labels, transform=test_transform)
        
        # Create dataloaders
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=shuffle_train,
            num_workers=workers,
            pin_memory=True,
            drop_last=True,
        )
        
        val_loader = DataLoader(
            test_dataset,
            batch_size=val_batch_size,
            shuffle=shuffle_val,
            num_workers=workers,
            pin_memory=True,
        )
        
        return train_loader, val_loader


def load_filtered_dataset(dataset_dir: str) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    """
    Load a filtered dataset from disk.
    
    This is a convenience function that loads images, labels, and metadata
    from a filtered dataset directory.
    
    Args:
        dataset_dir: Path to the filtered dataset directory
        
    Returns:
        Tuple of (images, labels, metadata)
    """
    dataset_path = Path(dataset_dir)
    
    images = np.load(dataset_path / "images.npy")
    labels = np.load(dataset_path / "labels.npy")
    
    try:
        metadata = np.load(dataset_path / "metadata.npy", allow_pickle=True).item()
    except FileNotFoundError:
        metadata = {}
    
    return images, labels, metadata


def load_natural_dataset(data_dir: str, split: str = "train") -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    """
    Load natural CIFAR-10 dataset from disk.
    
    Args:
        data_dir: Path to the natural dataset directory
        split: "train" or "test"
        
    Returns:
        Tuple of (images, labels, metadata)
    """
    data_path = Path(data_dir)
    prefix = f"cifar10_{split}"
    
    images = np.load(data_path / f"{prefix}_images.npy")
    labels = np.load(data_path / f"{prefix}_labels.npy")
    
    try:
        metadata = np.load(data_path / f"{prefix}_metadata.npy", allow_pickle=True).item()
    except FileNotFoundError:
        metadata = {}
    
    return images, labels, metadata


if __name__ == "__main__":
    """
    Test the dataset classes.
    """
    print("=" * 60)
    print("Testing FilteredCIFAR10 Dataset")
    print("=" * 60)
    
    # Test with natural dataset
    data_path = "./data/cifar10_natural"
    
    if Path(data_path).exists():
        print(f"\nLoading natural dataset from {data_path}...")
        ds = FilteredCIFAR10(data_path, dataset_type="natural")
        train_loader, val_loader = ds.make_loaders(batch_size=32, workers=2)
        
        print(f"Train batches: {len(train_loader)}")
        print(f"Val batches: {len(val_loader)}")
        
        # Test a batch
        images, labels = next(iter(train_loader))
        print(f"Batch shape: {images.shape}")
        print(f"Labels shape: {labels.shape}")
        print(f"Image range: [{images.min():.3f}, {images.max():.3f}]")
    else:
        print(f"Dataset not found at {data_path}. Run the pipeline first.")

