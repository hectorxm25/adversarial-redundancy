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
├── validation/
│   ├── variance_test.py             # Direct pixel-wise variance validation
│   └── power_spectrum.py            # Fourier power spectrum analysis
├── experiment_scripts/
│   ├── hyperparam_search_dft.sh      # Run pipeline with multiple radii
│   ├── visualize_first_images.sh    # Visualize single images from datasets
│   ├── visualize_comparison.sh      # Visualize multiple images for comparison
│   └── run_validation.sh            # Run both validation methods
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

### `validation/variance_test.py`

Direct pixel-wise variance test to validate filtered datasets.

**Main Function:**

- `run_variance_test()`: Computes and compares variance across datasets
  - Calculates pixel-wise variance for original dataset (baseline)
  - Calculates variance for high-variance (low-pass) dataset
  - Calculates variance for low-variance (high-pass) dataset
  - Reports percentage of variance captured by each
  - Saves results to text file

**Command-line Arguments:**

- `--natural-dataset-dir`: Path to original CIFAR-10 (default: `./data/cifar10_natural`)
- `--filtered-dataset-dir`: Path to filtered datasets (default: `./data/filtered_r10`)
- `--cutoff-radius`: Cutoff radius used for filtering (default: 10)
- `--n-samples`: Number of samples to use (default: 1000)
- `--output-dir`: Directory for results (default: `./validation_results`)

**Example Usage:**

```bash
python3 variance_test.py \
  --natural-dataset-dir ./data/cifar10_natural \
  --filtered-dataset-dir ./data/filtered_r10 \
  --cutoff-radius 10 \
  --n-samples 1000
```

### `validation/power_spectrum.py`

Fourier power spectrum analysis (recreates paper's Figure A.12).

**Main Function:**

- `create_power_spectrum_plot()`: Creates power spectrum visualization
  - Computes 2D FFT for each image and channel
  - Calculates power (squared magnitude) of frequency components
  - Averages across images and sorts by power
  - Creates plot showing sharp "elbow" and "long tail"
  - Marks cutoff radii on the plot
  - Saves plot and numerical results

**Command-line Arguments:**

- `--natural-dataset-dir`: Path to original CIFAR-10 (default: `./data/cifar10_natural`)
- `--n-samples`: Number of samples (default: 50, sufficient per paper)
- `--output-dir`: Directory for results (default: `./validation_results`)
- `--cutoff-radii`: Cutoff radii to mark on plot (default: [5, 10, 15])

**Example Usage:**

```bash
python3 power_spectrum.py \
  --natural-dataset-dir ./data/cifar10_natural \
  --n-samples 50 \
  --cutoff-radii 5 10 15
```

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

## Dataset Validation

Before training models, it's critical to verify that the filtered datasets correctly separate high-variance and low-variance components. We provide two validation methods based on the research methodology:

### Quick Start: Run All Validations

```bash
# Run both validation methods automatically
bash experiment_scripts/run_validation.sh
```

This will generate:
- Power spectrum plot showing frequency distribution
- Variance test results for each cutoff radius

### Direct Variance Test

Compute pixel-wise variance and verify that:
- High-variance (low-pass) dataset captures **>80%** of original variance
- Low-variance (high-pass) dataset captures **<20%** of original variance

```bash
cd validation

# Test a specific cutoff radius
python3 variance_test.py \
  --natural-dataset-dir ../data/cifar10_natural \
  --filtered-dataset-dir ../data/filtered_r10 \
  --cutoff-radius 10 \
  --n-samples 1000 \
  --output-dir ../validation_results
```

**What it does:**
1. Loads 1000 images from original and filtered datasets
2. Calculates total pixel-wise variance for each
3. Reports percentage of variance captured by each filtered dataset
4. Saves results to `validation_results/variance_test_r{radius}.txt`

**Expected Results:**
```
Original dataset variance:       100.00%
High-variance dataset variance:   ~85-95%  ✓
Low-variance dataset variance:    ~5-15%   ✓
```

### Power Spectrum Plot

Creates a Fourier power spectrum plot (recreates Figure A.12 from the paper) showing:
- Sharp "elbow": Variance concentrated in low-frequency components
- Long "tail": Minimal variance in high-frequency components

```bash
cd validation

# Create power spectrum plot
python3 power_spectrum.py \
  --natural-dataset-dir ../data/cifar10_natural \
  --n-samples 50 \
  --output-dir ../validation_results \
  --cutoff-radii 5 10 15
```

**What it does:**
1. Loads 50 images from original dataset
2. Computes 2D FFT for each image and channel
3. Calculates power spectrum (squared magnitude)
4. Averages across all images and sorts by power
5. Creates plot with marked cutoff radii
6. Saves to `validation_results/power_spectrum_plot.png`

**Interpretation:**
The plot validates the filtering approach by showing that most variance is concentrated in the first few frequency components. This justifies:
- Low-pass filters isolate the "head" (high-variance components)
- High-pass filters isolate the "tail" (low-variance components)

### Using Validation Results

After running validation:

1. **Check Power Spectrum Plot**: Confirm the sharp elbow and long tail
2. **Review Variance Tests**: Ensure proper variance separation for each radius
3. **Choose Optimal Radius**: Select the cutoff that best separates variance while maintaining enough information in each dataset
4. **Proceed to Training**: Use validated datasets for model training

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

### Completed Tasks

✅ **Dataset Creation**: Pipeline for creating filtered datasets with DFT  
✅ **Visualization**: Tools for visualizing filtered images  
✅ **Validation**: Both variance test and power spectrum analysis implemented

### Immediate Tasks

1. **Run Validation**: Execute `bash experiment_scripts/run_validation.sh` to verify datasets
2. **Model Training**: Implement CNN (ResNet-18) training on filtered datasets
3. **Ensemble Creation**: Implement voting schemes for combining model predictions
4. **Hyperparameter Tuning**: Use validation results to select optimal cutoff radius

### Advanced Tasks

1. **Adversarial Robustness Testing**:
   - Implement PGD (Projected Gradient Descent) attacks
   - Implement FGSM (Fast Gradient Sign Method) attacks
   - Evaluate ensemble robustness vs. single model

2. **Alternative Filters** (Time Permitting):
   - Implement Wavelet transform filtering
   - Compare DFT with full PCA implementation
   - Reproduce paper results with multiple filtering methods

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