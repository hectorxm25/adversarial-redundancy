"""
Redundancy Ensemble implementation combining natural, high-variance, and low-variance models.

This module provides the core ensemble class that implements the redundancy-based
approach to adversarial robustness by leveraging frequency-domain separation.
"""

import sys
from pathlib import Path
from typing import Optional, Tuple, Literal
from collections import Counter

import numpy as np
import torch
import torch.nn as nn
from torchvision import transforms

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from robustness import model_utils
from train.datasets import FilteredCIFAR10, load_natural_dataset, load_filtered_dataset
from filters.dft import create_butterworth_mask


class NoClearWinner(Exception):
    """
    Exception raised when the ensemble cannot determine a clear winner.
    
    This occurs in Strong voting mode when all three models do not agree
    on the same class, or when there is no clear majority.
    
    Attributes:
        predictions: Tuple of (natural_pred, high_var_pred, low_var_pred)
        message: Description of why no winner was determined
    """
    
    def __init__(
        self,
        predictions: Tuple[int, int, int],
        message: str = "No clear winner in ensemble voting"
    ):
        self.predictions = predictions
        self.message = message
        super().__init__(f"{message}: predictions = {predictions}")


class RedundancyEnsemble(nn.Module):
    """
    Ensemble model combining natural, high-variance, and low-variance models.
    
    This ensemble implements the redundancy-based approach to adversarial robustness
    by processing input images through three parallel pathways:
    
    1. Natural pathway: Input image → Natural model → Prediction
    2. High-variance pathway: Input image → Low-pass filter → High-variance model → Prediction
    3. Low-variance pathway: Input image → High-pass filter → Low-variance model → Prediction
    
    The final prediction is determined by voting:
    - Weak mode: Majority vote (at least 2 out of 3 agree)
    - Strong mode: Unanimous vote (all 3 must agree)
    
    Attributes:
        cutoff_radius: The radial frequency cutoff used for filtering
        voting_mode: Either 'weak' (majority) or 'strong' (unanimous)
        natural_model: Model trained on natural images
        high_var_model: Model trained on low-pass (high-variance) filtered images
        low_var_model: Model trained on high-pass (low-variance) filtered images
    """
    
    # CIFAR-10 standard statistics for normalization
    CIFAR10_MEAN = torch.tensor([0.4914, 0.4822, 0.4465])
    CIFAR10_STD = torch.tensor([0.2470, 0.2435, 0.2616])
    
    def __init__(
        self,
        natural_model_path: str,
        high_var_model_path: str,
        low_var_model_path: str,
        cutoff_radius: float,
        voting_mode: Literal['weak', 'strong'] = 'weak',
        butterworth_order: int = 2,
        device: Optional[str] = None,
    ):
        """
        Initialize the RedundancyEnsemble.
        
        Args:
            natural_model_path: Path to the natural model checkpoint
            high_var_model_path: Path to the high-variance model checkpoint
            low_var_model_path: Path to the low-variance model checkpoint
            cutoff_radius: Radial frequency cutoff for DFT filtering (e.g., 5, 10, 15)
            voting_mode: 'weak' for majority vote, 'strong' for unanimous vote
            butterworth_order: Order of the Butterworth filter (default: 2)
            device: Device to use (e.g., 'cuda:0', 'cpu'). If None, auto-selects.
        """
        super().__init__()
        
        self.cutoff_radius = cutoff_radius
        self.voting_mode = voting_mode
        self.butterworth_order = butterworth_order
        
        # Determine device
        if device is None:
            self._device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        else:
            self._device = torch.device(device)
        
        # Create dummy dataset for model loading (required by robustness library)
        # We don't need actual data, just the dataset structure
        self._dummy_dataset = self._create_dummy_dataset()
        
        # Load models
        print(f"Loading natural model from: {natural_model_path}")
        self.natural_model = self._load_model(natural_model_path)
        
        print(f"Loading high-variance model from: {high_var_model_path}")
        self.high_var_model = self._load_model(high_var_model_path)
        
        print(f"Loading low-variance model from: {low_var_model_path}")
        self.low_var_model = self._load_model(low_var_model_path)
        
        # Pre-compute Butterworth masks for efficiency (for 32x32 CIFAR-10 images)
        self._low_pass_mask = self._create_mask(high_pass=False)
        self._high_pass_mask = self._create_mask(high_pass=True)
        
        # Move masks to device
        self._low_pass_mask = self._low_pass_mask.to(self._device)
        self._high_pass_mask = self._high_pass_mask.to(self._device)
        
        # Set models to eval mode
        self.natural_model.eval()
        self.high_var_model.eval()
        self.low_var_model.eval()
        
        print(f"Ensemble initialized with cutoff_radius={cutoff_radius}, voting_mode={voting_mode}")
        print(f"Device: {self._device}")
    
    def _create_dummy_dataset(self) -> FilteredCIFAR10:
        """Create a dummy dataset for model loading."""
        # Create minimal dummy data
        dummy_images = np.zeros((1, 32, 32, 3), dtype=np.uint8)
        dummy_labels = np.zeros(1, dtype=np.int64)
        
        return FilteredCIFAR10(
            data_path="/tmp",  # Dummy path
            dataset_type="natural",
            train_images=dummy_images,
            train_labels=dummy_labels,
            test_images=dummy_images,
            test_labels=dummy_labels,
        )
    
    def _load_model(self, checkpoint_path: str) -> nn.Module:
        """
        Load a model from a checkpoint.
        
        Args:
            checkpoint_path: Path to the model checkpoint
            
        Returns:
            Loaded model on the specified device
        """
        model, _ = model_utils.make_and_restore_model(
            arch='resnet18',
            dataset=self._dummy_dataset,
            resume_path=checkpoint_path,
            parallel=False,
        )
        
        # Move model to device
        model = model.to(self._device)
        model.eval()
        
        return model
    
    def _create_mask(self, high_pass: bool = False) -> torch.Tensor:
        """
        Create a Butterworth filter mask.
        
        Args:
            high_pass: If True, create high-pass mask; otherwise low-pass
            
        Returns:
            Filter mask as a torch tensor
        """
        # CIFAR-10 images are 32x32
        mask = create_butterworth_mask(
            shape=(32, 32),
            cutoff_radius=self.cutoff_radius,
            order=self.butterworth_order,
            high_pass=high_pass,
        )
        return torch.from_numpy(mask).float()
    
    def _apply_dft_filter(
        self,
        images: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Apply DFT filtering to a batch of images.
        
        This applies the frequency-domain filter to each channel of each image
        in the batch using PyTorch's FFT operations for GPU acceleration.
        
        Args:
            images: Batch of images, shape (B, C, H, W), normalized
            mask: Frequency mask, shape (H, W)
            
        Returns:
            Filtered images, same shape as input
        """
        batch_size, channels, height, width = images.shape
        
        # Denormalize images to [0, 1] range for filtering
        mean = self.CIFAR10_MEAN.view(1, 3, 1, 1).to(images.device)
        std = self.CIFAR10_STD.view(1, 3, 1, 1).to(images.device)
        images_denorm = images * std + mean
        images_denorm = torch.clamp(images_denorm, 0, 1)
        
        # Process each channel with FFT
        filtered = torch.zeros_like(images_denorm)
        
        for c in range(channels):
            channel = images_denorm[:, c, :, :]  # (B, H, W)
            
            # Apply 2D FFT
            fft_channel = torch.fft.fft2(channel)
            
            # Shift zero-frequency to center
            fft_shifted = torch.fft.fftshift(fft_channel, dim=(-2, -1))
            
            # Apply mask (broadcast over batch dimension)
            filtered_fft_shifted = fft_shifted * mask.unsqueeze(0)
            
            # Inverse shift
            filtered_fft = torch.fft.ifftshift(filtered_fft_shifted, dim=(-2, -1))
            
            # Inverse FFT and take real part
            filtered_channel = torch.fft.ifft2(filtered_fft).real
            
            filtered[:, c, :, :] = filtered_channel
        
        # Clip to valid range
        filtered = torch.clamp(filtered, 0, 1)
        
        # Re-normalize to match model input expectations
        filtered_normalized = (filtered - mean) / std
        
        return filtered_normalized
    
    def _get_predictions(
        self,
        images: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Get predictions from all three models.
        
        Args:
            images: Batch of normalized images, shape (B, C, H, W)
            
        Returns:
            Tuple of (natural_logits, high_var_logits, low_var_logits)
        """
        # Natural model: use original images directly
        natural_output, _ = self.natural_model(images)
        
        # High-variance model: apply low-pass filter first
        high_var_images = self._apply_dft_filter(images, self._low_pass_mask)
        high_var_output, _ = self.high_var_model(high_var_images)
        
        # Low-variance model: apply high-pass filter first
        low_var_images = self._apply_dft_filter(images, self._high_pass_mask)
        low_var_output, _ = self.low_var_model(low_var_images)
        
        return natural_output, high_var_output, low_var_output
    
    def forward(
        self,
        images: torch.Tensor,
        return_all_predictions: bool = False,
    ) -> torch.Tensor:
        """
        Forward pass through the ensemble.
        
        Args:
            images: Batch of normalized images, shape (B, C, H, W)
            return_all_predictions: If True, return all three model predictions
                                    instead of the voted result
            
        Returns:
            If return_all_predictions is False:
                Voted predictions, shape (B,)
            If return_all_predictions is True:
                Tuple of (voted_predictions, natural_preds, high_var_preds, low_var_preds)
                
        Raises:
            NoClearWinner: In 'strong' mode when all three models don't agree
        """
        # Get predictions from all models
        natural_logits, high_var_logits, low_var_logits = self._get_predictions(images)
        
        # Get class predictions
        natural_preds = natural_logits.argmax(dim=1)
        high_var_preds = high_var_logits.argmax(dim=1)
        low_var_preds = low_var_logits.argmax(dim=1)
        
        # Vote on predictions
        voted_preds = self._vote(natural_preds, high_var_preds, low_var_preds)
        
        if return_all_predictions:
            return voted_preds, natural_preds, high_var_preds, low_var_preds
        return voted_preds
    
    def _vote(
        self,
        natural_preds: torch.Tensor,
        high_var_preds: torch.Tensor,
        low_var_preds: torch.Tensor,
    ) -> torch.Tensor:
        """
        Vote on predictions based on the voting mode.
        
        Args:
            natural_preds: Predictions from natural model, shape (B,)
            high_var_preds: Predictions from high-variance model, shape (B,)
            low_var_preds: Predictions from low-variance model, shape (B,)
            
        Returns:
            Voted predictions, shape (B,)
            
        Raises:
            NoClearWinner: In 'strong' mode when not all predictions match
        """
        batch_size = natural_preds.shape[0]
        voted = torch.zeros_like(natural_preds)
        
        for i in range(batch_size):
            nat = natural_preds[i].item()
            high = high_var_preds[i].item()
            low = low_var_preds[i].item()
            
            predictions = (nat, high, low)
            
            if self.voting_mode == 'strong':
                # Strong mode: all three must agree
                if nat == high == low:
                    voted[i] = nat
                else:
                    raise NoClearWinner(
                        predictions=predictions,
                        message="Strong voting requires all three models to agree"
                    )
            else:
                # Weak mode: majority vote
                counter = Counter(predictions)
                most_common = counter.most_common(1)[0]
                
                if most_common[1] >= 2:
                    # At least 2 out of 3 agree
                    voted[i] = most_common[0]
                else:
                    # All three disagree (edge case: shouldn't happen with 3 classifiers)
                    raise NoClearWinner(
                        predictions=predictions,
                        message="No majority in weak voting (all three disagree)"
                    )
        
        return voted
    
    def predict_with_logits(
        self,
        images: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Get predictions along with raw logits from all models.
        
        Args:
            images: Batch of normalized images, shape (B, C, H, W)
            
        Returns:
            Tuple of (voted_preds, natural_logits, high_var_logits, low_var_logits)
        """
        natural_logits, high_var_logits, low_var_logits = self._get_predictions(images)
        
        natural_preds = natural_logits.argmax(dim=1)
        high_var_preds = high_var_logits.argmax(dim=1)
        low_var_preds = low_var_logits.argmax(dim=1)
        
        voted_preds = self._vote(natural_preds, high_var_preds, low_var_preds)
        
        return voted_preds, natural_logits, high_var_logits, low_var_logits
    
    def get_individual_accuracies(
        self,
        images: torch.Tensor,
        labels: torch.Tensor,
    ) -> Tuple[float, float, float, float]:
        """
        Compute accuracy for each model and the ensemble.
        
        Args:
            images: Batch of normalized images, shape (B, C, H, W)
            labels: True labels, shape (B,)
            
        Returns:
            Tuple of (ensemble_acc, natural_acc, high_var_acc, low_var_acc)
        """
        with torch.no_grad():
            voted_preds, natural_preds, high_var_preds, low_var_preds = self.forward(
                images, return_all_predictions=True
            )
        
        ensemble_acc = (voted_preds == labels).float().mean().item() * 100
        natural_acc = (natural_preds == labels).float().mean().item() * 100
        high_var_acc = (high_var_preds == labels).float().mean().item() * 100
        low_var_acc = (low_var_preds == labels).float().mean().item() * 100
        
        return ensemble_acc, natural_acc, high_var_acc, low_var_acc


def load_ensemble(
    models_dir: str,
    cutoff_radius: int,
    voting_mode: Literal['weak', 'strong'] = 'weak',
    device: Optional[str] = None,
) -> RedundancyEnsemble:
    """
    Convenience function to load an ensemble from a models directory.
    
    Args:
        models_dir: Directory containing trained model checkpoints
        cutoff_radius: Cutoff radius (5, 10, or 15)
        voting_mode: 'weak' for majority vote, 'strong' for unanimous vote
        device: Device to use (e.g., 'cuda:0', 'cpu')
        
    Returns:
        Loaded RedundancyEnsemble
    """
    models_path = Path(models_dir)
    
    # Find checkpoint files
    natural_checkpoint = models_path / "resnet18_natural" / "checkpoint.pt.best"
    if not natural_checkpoint.exists():
        natural_checkpoint = models_path / "resnet18_natural" / "checkpoint.pt"
    
    high_var_checkpoint = models_path / f"resnet18_high_variance_r{cutoff_radius}" / "checkpoint.pt.best"
    if not high_var_checkpoint.exists():
        high_var_checkpoint = models_path / f"resnet18_high_variance_r{cutoff_radius}" / "checkpoint.pt"
    
    low_var_checkpoint = models_path / f"resnet18_low_variance_r{cutoff_radius}" / "checkpoint.pt.best"
    if not low_var_checkpoint.exists():
        low_var_checkpoint = models_path / f"resnet18_low_variance_r{cutoff_radius}" / "checkpoint.pt"
    
    # Validate checkpoints exist
    for checkpoint, name in [
        (natural_checkpoint, "natural"),
        (high_var_checkpoint, f"high_variance_r{cutoff_radius}"),
        (low_var_checkpoint, f"low_variance_r{cutoff_radius}"),
    ]:
        if not checkpoint.exists():
            raise FileNotFoundError(f"Checkpoint not found for {name}: {checkpoint}")
    
    return RedundancyEnsemble(
        natural_model_path=str(natural_checkpoint),
        high_var_model_path=str(high_var_checkpoint),
        low_var_model_path=str(low_var_checkpoint),
        cutoff_radius=float(cutoff_radius),
        voting_mode=voting_mode,
        device=device,
    )

