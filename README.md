# Adversarial Redundancy

A research project exploring adversarial robustness through redundant ensemble learning using frequency-domain filtering. This pipeline implements 2D Discrete Fourier Transform (DFT) filtering to separate high-variance (low-frequency) and low-variance (high-frequency) components of images, creating complementary datasets for ensemble training.

## Overview

This project implements the frequency-domain filtering approach outlined in research showing that under translational invariance assumptions, the Fourier basis is equivalent to the PCA basis. By filtering images in the frequency domain, we can create two complementary views of the data:

- **High-Variance Dataset** (Low-Pass Filter): Retains low-frequency components that capture global structure
- **Low-Variance Dataset** (High-Pass Filter): Retains high-frequency components that capture fine details

These datasets can then be used to train an ensemble of models that leverage different types of information, potentially improving adversarial robustness.

## Project Structure

```
adversarial-redundancy/
├── filters/
│   ├── dft.py              # 2D DFT filtering implementation
│   ├── utils.py            # CIFAR-10 loading and data management utilities
│   ├── pipeline.py         # Complete end-to-end pipeline script
│   └── __init__.py         # (optional) Package initialization
├── experiment_scripts/
│   ├── hyperparam_search_dft.sh      # Run pipeline with multiple radii
│   ├── visualize_first_images.sh    # Visualize single images from datasets
│   └── visualize_comparison.sh      # Visualize multiple images for comparison
├── help.txt            # Detailed implementation guide and references
└── README.md           # This file
```

## Installation

### Requirements

- Python 3.8+
- PyTorch
- torchvision
- NumPy
- tqdm
- matplotlib (for visualization)

### Setup

```bash
# Install dependencies
pip install torch torchvision numpy tqdm matplotlib

# Or with conda
conda install pytorch torchvision numpy tqdm matplotlib -c pytorch
```

## Usage

### Quick Start (Using Pipeline Script - Recommended)

The easiest way to get started is using the automated pipeline script:

```bash
# Navigate to filters directory
cd filters

# Run pipeline with default settings
python pipeline.py

# Custom directories and parameters
python pipeline.py \
  --natural-dataset-dir ./my_data/natural \
  --filtered-dataset-dir ./my_data/filtered \
  --cutoff-radius 10

# Experiment with different cutoff radii
python pipeline.py --cutoff-radius 5 --filtered-dataset-dir ./data/filtered_r5
python pipeline.py --cutoff-radius 10 --filtered-dataset-dir ./data/filtered_r10
python pipeline.py --cutoff-radius 15 --filtered-dataset-dir ./data/filtered_r15
```

This will automatically:
1. Download CIFAR-10 to the natural dataset directory
2. Create high-variance (low-pass) and low-variance (high-pass) filtered datasets
3. Process both training and test sets
4. Save everything to the specified directories

### Programmatic Usage

For more control, use the modules directly in Python:

```python
from filters.utils import load_cifar10, save_cifar10
from filters.dft import process_dataset

# 1. Load CIFAR-10 dataset
train_dataset = load_cifar10(root="./data", train=True, download=True)
test_dataset = load_cifar10(root="./data", train=False, download=True)

# 2. Process dataset with DFT filtering
high_var_dir, low_var_dir = process_dataset(
    dataset=train_dataset,
    output_dir="./filtered_data",
    cutoff_radius=10,  # Key hyperparameter to tune
    use_butterworth=True,  # Recommended to avoid ringing
    butterworth_order=2,
    dataset_name="cifar10_train"
)

# 3. Load filtered datasets for training
from filters.dft import load_filtered_dataset

high_var_images, high_var_labels, metadata = load_filtered_dataset(high_var_dir)
low_var_images, low_var_labels, metadata = load_filtered_dataset(low_var_dir)
```

### Running the Examples

**Production Pipeline:**

```bash
cd filters

# Run the complete pipeline (recommended for actual use)
python pipeline.py

# See all available options
python pipeline.py --help
```

**Testing Individual Modules:**

Both `dft.py` and `utils.py` include example usage in their `__main__` blocks:

```bash
cd filters

# Test DFT filtering pipeline
python dft.py

# Test data utilities
python utils.py
```

## Module Documentation

### `filters/pipeline.py`

Complete end-to-end pipeline script that orchestrates the entire workflow.

**Main Function:**

- `run_pipeline()`: Executes the complete pipeline with configurable parameters
  - Downloads CIFAR-10 to specified directory
  - Saves natural datasets for faster reloading
  - Processes both training and test sets with DFT filtering
  - Creates high-variance and low-variance datasets
  - Saves all results to specified output directory

**Command-line Arguments:**

- `--natural-dataset-dir`: Directory for original CIFAR-10 (default: `./data/cifar10_natural`)
- `--filtered-dataset-dir`: Directory for filtered datasets (default: `./data/cifar10_filtered`)
- `--cutoff-radius`: Radial frequency cutoff (default: 10.0)
- `--use-butterworth/--no-butterworth`: Filter type selection (default: Butterworth)
- `--butterworth-order`: Butterworth filter order (default: 2)
- `--no-download`: Skip download if dataset exists

**Example Usage:**

```bash
# Basic usage
python pipeline.py

# Custom configuration
python pipeline.py \
  --natural-dataset-dir ./data/natural \
  --filtered-dataset-dir ./data/filtered \
  --cutoff-radius 12 \
  --butterworth-order 3
```

### `filters/dft.py`

Implements 2D Discrete Fourier Transform filtering for image processing.

**Key Functions:**

- `create_butterworth_mask()`: Creates smooth frequency masks to avoid ringing artifacts
- `create_circular_mask()`: Creates hard circular masks (alternative to Butterworth)
- `apply_dft_filter_to_image()`: Applies frequency-domain filtering to a single image
- `process_dataset()`: Batch processes entire dataset and saves filtered versions
- `load_filtered_dataset()`: Loads previously filtered datasets from disk

**Key Features:**

- Per-channel (R, G, B) 2D FFT processing
- Butterworth filter for smooth transitions (reduces ringing effects)
- Support for both low-pass and high-pass filtering
- Efficient batch processing with progress bars
- Automatic data normalization and type handling

### `filters/utils.py`

Provides utilities for CIFAR-10 dataset management.

**Key Functions:**

- `load_cifar10()`: Loads CIFAR-10 using torchvision
- `save_cifar10()`: Saves CIFAR-10 to disk in numpy format for faster loading
- `load_saved_cifar10()`: Loads previously saved numpy datasets
- `create_dataloader()`: Creates PyTorch DataLoader with standard settings
- `get_cifar10_statistics()`: Computes mean/std for normalization
- `get_standard_transforms()`: Provides standard augmentation and normalization
- `visualize_images()`: Visualizes and saves the first N images from a dataset

**Visualization:**

```bash
# Visualize images from a filtered dataset
python3 utils.py \
  --input-filepath ./data/filtered_r10/cifar10_train_high_variance_r10 \
  --output-filepath ./visualizations/high_var.png \
  --n-images 10 \
  --dataset-type numpy

# Visualize natural CIFAR-10 images
python3 utils.py \
  --input-filepath ./data/cifar10_natural \
  --output-filepath ./visualizations/natural.png \
  --n-images 10 \
  --dataset-type cifar10
```

**Dataset Information:**

- 60,000 32×32 color images across 10 classes
- 50,000 training images, 10,000 test images
- Classes: airplane, automobile, bird, cat, deer, dog, frog, horse, ship, truck

## Implementation Details

### 2D DFT Filtering Pipeline

For each image in the dataset:

1. **Channel Separation**: Split RGB image into separate channels (R, G, B)
2. **2D FFT**: Apply Fast Fourier Transform to each channel
   - NumPy: `np.fft.fft2()`
   - PyTorch: `torch.fft.fft2()`
3. **Frequency Shift**: Move zero-frequency (DC) component to center
   - `np.fft.fftshift()` or `torch.fft.fftshift()`
4. **Apply Mask**: Element-wise multiplication with frequency mask
   - Low-pass mask: Keeps frequencies within radius `r` from center (high-variance)
   - High-pass mask: Keeps frequencies beyond radius `r` from center (low-variance)
   - Butterworth filter: Smooth transition to reduce ringing artifacts
5. **Inverse Operations**: Unshift, inverse FFT, take real part
6. **Recombine**: Stack filtered channels back into RGB image

### Hyperparameter Tuning

The cutoff radius `r` is a critical hyperparameter that determines the frequency separation:

- **Small `r` (e.g., 5)**: Aggressive filtering, more separation between datasets
- **Medium `r` (e.g., 10)**: Balanced filtering (recommended starting point)
- **Large `r` (e.g., 15)**: Conservative filtering, less separation

Recommended approach:
```python
# Experiment with different cutoff radii
for r in [5, 8, 10, 12, 15]:
    process_dataset(train_dataset, f"./filtered_r{r}", cutoff_radius=r)
```

### Butterworth vs. Circular Filters

**Butterworth Filter** (Recommended):
- Smooth transition between pass and stop bands
- Reduces ringing artifacts
- Formula: `H(u,v) = 1 / (1 + (D(u,v)/D0)^(2n))`
- Use `use_butterworth=True` in `process_dataset()`

**Circular Filter**:
- Hard cutoff (binary mask)
- Can create ringing effects in filtered images
- Simpler but less clean results
- Use `use_butterworth=False` if needed for comparison

## Visualization

After creating filtered datasets, visualize them to understand the filtering effects:

### Quick Visualization (Single Images)

```bash
# Visualize first image from each dataset for quick comparison
bash experiment_scripts/visualize_first_images.sh
```

This creates single-image visualizations:
- `cifar10_natural_first_image.png`
- `cifar10_train_high_variance_r{5,10,15}_first_image.png`
- `cifar10_train_low_variance_r{5,10,15}_first_image.png`

### Comprehensive Visualization (10 Images)

```bash
# Visualize 10 images from each dataset for thorough comparison
bash experiment_scripts/visualize_comparison.sh
```

This creates multi-image grids showing the effects of different cutoff radii on various image types.

### Custom Visualization

```bash
# Visualize specific dataset with custom parameters
cd filters
python3 utils.py \
  --input-filepath ../data/filtered_r10/cifar10_train_high_variance_r10 \
  --output-filepath ../visualizations/my_custom_viz.png \
  --n-images 20 \
  --dataset-type numpy \
  --grid-cols 5
```

## Workflow Example

### Using Pipeline Script (Simplest Method)

For most use cases, simply use the provided pipeline script:

```bash
# Process with single cutoff radius
python filters/pipeline.py --cutoff-radius 10

# Experiment with multiple radii (bash loop)
for r in 5 10 15; do
  python filters/pipeline.py \
    --cutoff-radius $r \
    --filtered-dataset-dir ./data/filtered_r${r}
done
```

### Complete Pipeline (Programmatic)

For advanced use cases requiring custom logic:

```python
from filters.utils import load_cifar10, save_cifar10, create_dataloader
from filters.dft import process_dataset, load_filtered_dataset
import torch

# 1. Data Loading
print("Loading CIFAR-10...")
train_dataset = load_cifar10(root="./data", train=True, download=True)
test_dataset = load_cifar10(root="./data", train=False, download=True)

# 2. Save raw data (optional, for faster reloading)
print("Saving raw data...")
save_cifar10(train_dataset, "./saved_data", "cifar10_train")
save_cifar10(test_dataset, "./saved_data", "cifar10_test")

# 3. Create filtered datasets with multiple radii
print("Creating filtered datasets...")
cutoff_radii = [5, 10, 15]

for r in cutoff_radii:
    # Process training set
    train_high_var, train_low_var = process_dataset(
        train_dataset,
        output_dir=f"./filtered_data_r{r}",
        cutoff_radius=r,
        use_butterworth=True,
        dataset_name="cifar10_train"
    )
    
    # Process test set
    test_high_var, test_low_var = process_dataset(
        test_dataset,
        output_dir=f"./filtered_data_r{r}",
        cutoff_radius=r,
        use_butterworth=True,
        dataset_name="cifar10_test"
    )

# 4. Load filtered data for training
print("Loading filtered data for training...")
high_var_images, high_var_labels, metadata = load_filtered_dataset(train_high_var)
low_var_images, low_var_labels, metadata = load_filtered_dataset(train_low_var)

# 5. Create dataloaders
from filters.utils import CIFAR10Dataset

high_var_dataset = CIFAR10Dataset(high_var_images, high_var_labels)
low_var_dataset = CIFAR10Dataset(low_var_images, low_var_labels)

high_var_loader = create_dataloader(high_var_dataset, batch_size=128)
low_var_loader = create_dataloader(low_var_dataset, batch_size=128)

# 6. Train models on each dataset
# TODO: Implement model training
# model_high_var = train_model(high_var_loader)
# model_low_var = train_model(low_var_loader)

# 7. Create ensemble and test adversarial robustness
# TODO: Implement ensemble and adversarial testing
# ensemble = Ensemble([model_high_var, model_low_var])
# test_adversarial_robustness(ensemble, test_loader)
```

## Next Steps

### Immediate Tasks

0. **Scree Plot**: Verify the variance of the filtered dataset(s)
1. **Hyperparameter Search**: Systematically evaluate different cutoff radii
2. **Model Training**: Implement CNN training on filtered datasets
3. **Ensemble Creation**: Implement voting schemes for combining model predictions
4. **Extra Filters (Time Willing)**: Implement different filters (e.g. wavelet transform) to reproduce results

### Advanced Tasks

**Adversarial Robustness Testing**:
   - Implement PGD (Projected Gradient Descent) attacks
   - Implement FGSM (Fast Gradient Sign Method) attacks
   - Evaluate ensemble robustness vs. single model

## Research Background

This implementation is based on research showing that:

1. Under translational invariance assumptions (reasonable for natural images), the Fourier basis approximates the PCA basis
2. Separating high-variance and low-variance components creates complementary views of data
3. Training separate models on these complementary views and ensembling may improve adversarial robustness
4. The DFT approach is computationally efficient compared to computing full PCA on large datasets (avoids 3072×3072 covariance matrix computation for CIFAR-10)

## References

TODO: Fill in

## Citation

TODO: Fill in

## License

TODO: Fill in

## Contact

TODO: Fill in