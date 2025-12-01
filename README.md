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
│   └── __init__.py         # Package initialization
├── train/
│   ├── __init__.py         # Package initialization
│   ├── datasets.py         # Custom dataset wrappers for robustness library
│   └── train_models.py     # ResNet18 training pipeline using robustness library
├── evaluate/
│   ├── __init__.py         # Package initialization
│   └── evaluate_models.py  # Model evaluation pipeline for computing test accuracies
├── adversarial/
│   ├── __init__.py              # Package initialization
│   └── generate_adversarial.py  # Adversarial example generation using PGD and FGSM
├── validation/
│   ├── variance_test.py             # Direct pixel-wise variance validation
│   └── power_spectrum.py            # Fourier power spectrum analysis
├── experiment_scripts/
│   ├── hyperparam_search_dft.sh      # Run pipeline with multiple radii
│   ├── visualize_first_images.sh    # Visualize single images from datasets
│   ├── visualize_comparison.sh      # Visualize multiple images for comparison
│   ├── run_validation.sh            # Run both validation methods
│   ├── run_training.sh              # Train all models (natural + filtered)
│   ├── run_evaluation.sh            # Evaluate all trained models
│   └── run_adversarial.sh           # Generate adversarial examples for all models
├── data/                   # Dataset storage (created by pipeline)
│   ├── cifar10_natural/    # Original CIFAR-10 dataset
│   ├── filtered_r5/        # Filtered datasets with r=5
│   ├── filtered_r10/       # Filtered datasets with r=10
│   └── filtered_r15/       # Filtered datasets with r=15
├── models/                 # Trained model checkpoints (created by training)
├── evaluation_results/     # Evaluation results and reports (created by evaluation)
├── logs/                   # Training and experiment logs
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
- robustness (MadryLab library for adversarial training)
- Pillow (PIL)

### Setup

```bash
# Install core dependencies
pip install torch torchvision numpy tqdm matplotlib pillow

# Install the robustness library (required for training)
pip install git+https://github.com/MadryLab/robustness.git

# Or with conda (then install robustness via pip)
conda install pytorch torchvision numpy tqdm matplotlib pillow -c pytorch
pip install git+https://github.com/MadryLab/robustness.git
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

## Model Training

The training pipeline uses the [robustness library](https://github.com/MadryLab/robustness) from MadryLab to train ResNet18 models. This library is specifically designed for adversarial robustness research and makes it easy to generate adversarial examples later.

### Quick Start: Train All Models

```bash
# Train all models with default settings (100 epochs)
bash experiment_scripts/run_training.sh

# Train with custom epochs
bash experiment_scripts/run_training.sh --epochs 50

# Train only the natural model
bash experiment_scripts/run_training.sh --natural-only

# Train only filtered models
bash experiment_scripts/run_training.sh --filtered-only
```

This will train **7 models** total:
- 1 model on the natural CIFAR-10 dataset
- 3 models on high-variance (low-pass) filtered datasets (r=5, 10, 15)
- 3 models on low-variance (high-pass) filtered datasets (r=5, 10, 15)

### Training Output

After training completes, you'll find:

```
models/
├── resnet18_natural/
│   ├── checkpoint.pt           # Model checkpoint (robustness format)
│   └── training_stats.json     # Training statistics
├── resnet18_high_variance_r5/
│   ├── checkpoint.pt
│   └── training_stats.json
├── resnet18_high_variance_r10/
│   └── ...
├── resnet18_high_variance_r15/
│   └── ...
├── resnet18_low_variance_r5/
│   └── ...
├── resnet18_low_variance_r10/
│   └── ...
├── resnet18_low_variance_r15/
│   └── ...
└── training_results.json       # Summary of all training runs

logs/training/
└── training_pipeline_YYYYMMDD_HHMMSS.log  # Detailed training logs
```

### Command-Line Options

```bash
bash experiment_scripts/run_training.sh [OPTIONS]

Options:
  --natural-dir DIR       Path to natural CIFAR-10 dataset (default: ./data/cifar10_natural)
  --filtered-base-dir DIR Base path for filtered datasets (default: ./data)
  --output-dir DIR        Directory for model checkpoints (default: ./models)
  --log-dir DIR           Directory for training logs (default: ./logs/training)
  --epochs N              Number of training epochs (default: 100)
  --lr RATE               Initial learning rate (default: 0.1)
  --batch-size N          Training batch size (default: 128)
  --workers N             Number of data loading workers (default: 4)
  --log-iters N           How often to log progress (default: 100)
  --cutoff-radii "R1 R2"  Space-separated cutoff radii (default: "5 10 15")
  --natural-only          Only train on natural dataset
  --filtered-only         Only train on filtered datasets
  --no-natural            Skip natural dataset training
  --no-filtered           Skip filtered datasets training
  -h, --help              Show help message
```

### Programmatic Training

For custom training workflows:

```python
from train.train_models import train_single_model, run_training_pipeline
from train.datasets import FilteredCIFAR10, load_filtered_dataset, load_natural_dataset

# Option 1: Run the full pipeline
results = run_training_pipeline(
    natural_dir="./data/cifar10_natural",
    filtered_base_dir="./data",
    output_dir="./models",
    log_dir="./logs/training",
    epochs=100,
    lr=0.1,
    batch_size=128,
    cutoff_radii=[5, 10, 15],
)

# Option 2: Train a single model
train_images, train_labels, _ = load_natural_dataset("./data/cifar10_natural", "train")
test_images, test_labels, _ = load_natural_dataset("./data/cifar10_natural", "test")

dataset = FilteredCIFAR10(
    data_path="./data/cifar10_natural",
    dataset_type="natural",
    train_images=train_images,
    train_labels=train_labels,
    test_images=test_images,
    test_labels=test_labels,
)

checkpoint_path, stats = train_single_model(
    dataset=dataset,
    output_dir="./models",
    model_name="my_custom_model",
    epochs=50,
)
```

### Loading Trained Models

The trained models are saved in the robustness library's checkpoint format:

```python
from robustness import model_utils
from train.datasets import FilteredCIFAR10

# Create a dataset object (needed for model loading)
dataset = FilteredCIFAR10(
    data_path="./data/cifar10_natural",
    dataset_type="natural",
)

# Load the trained model
model, _ = model_utils.make_and_restore_model(
    arch='resnet18',
    dataset=dataset,
    resume_path='./models/resnet18_natural/checkpoint.pt',
)

# The model is now ready for inference or adversarial attacks
model.eval()
```

### GPU Memory Management

The training pipeline automatically manages GPU memory to support sequential training runs:

- After each model finishes training, the model weights are deleted from GPU memory
- `torch.cuda.empty_cache()` is called to free cached memory
- This ensures only one model occupies GPU memory at a time

This is critical when training all 7 models sequentially on a single GPU.

### Training Module Documentation

#### `train/datasets.py`

Custom dataset classes compatible with the robustness library:

- `NumpyDataset`: PyTorch Dataset wrapper for numpy arrays
- `FilteredCIFAR10`: Dataset class extending robustness's DataSet for our filtered datasets
- `load_filtered_dataset()`: Load filtered dataset from disk
- `load_natural_dataset()`: Load natural CIFAR-10 from disk

#### `train/train_models.py`

Main training script:

- `train_single_model()`: Train a single ResNet18 model on a dataset
- `run_training_pipeline()`: Run the complete training pipeline for all datasets
- `cleanup_gpu_memory()`: Clean up GPU memory after training
- `get_training_args()`: Create training arguments for the robustness library
- `setup_logging()`: Set up verbose logging for training

## Model Evaluation

After training models, you can evaluate them on test datasets to compute accuracy metrics. The evaluation pipeline can evaluate models on both natural and filtered test sets.

### Quick Start: Evaluate All Models

```bash
# Evaluate all trained models on natural and filtered test sets
bash experiment_scripts/run_evaluation.sh

# Evaluate only on natural test set
bash experiment_scripts/run_evaluation.sh --natural-test-only

# Evaluate only on filtered test sets
bash experiment_scripts/run_evaluation.sh --filtered-test-only

# Use specific GPU
bash experiment_scripts/run_evaluation.sh --device cuda:1
```

### Evaluation Output

After evaluation completes, you'll find:

```
evaluation_results/
└── evaluation_results.json    # Complete evaluation results

logs/evaluation/
└── evaluation_pipeline_YYYYMMDD_HHMMSS.log  # Detailed evaluation logs
```

The JSON file contains accuracy metrics for each model-dataset combination:
- Natural model evaluated on natural test set
- Filtered models evaluated on natural test set
- Filtered models evaluated on their corresponding filtered test sets

### Command-Line Options

```bash
bash experiment_scripts/run_evaluation.sh [OPTIONS]

Options:
  --models-dir DIR        Directory containing trained models (default: ./models)
  --natural-dir DIR       Path to natural CIFAR-10 dataset (default: ./data/cifar10_natural)
  --filtered-base-dir DIR Base path for filtered datasets (default: ./data)
  --output-dir DIR        Directory for evaluation results (default: ./evaluation_results)
  --log-dir DIR           Directory for evaluation logs (default: ./logs/evaluation)
  --batch-size N          Evaluation batch size (default: 128)
  --workers N             Number of data loading workers (default: 4)
  --device DEVICE         Device to use (e.g., 'cuda:0', 'cuda:1', 'cpu', or '0', '1' for GPU index)
  --cutoff-radii "R1 R2"  Space-separated cutoff radii (default: "5 10 15")
  --natural-only          Only evaluate natural model
  --filtered-only         Only evaluate filtered models
  --no-natural            Skip evaluating natural model
  --no-filtered           Skip evaluating filtered models
  --natural-test-only     Only evaluate on natural test set
  --filtered-test-only    Only evaluate on filtered test sets
  --no-natural-test       Skip evaluation on natural test set
  --no-filtered-test      Skip evaluation on filtered test sets
  --adv-eval              Perform adversarial evaluation (in addition to natural)
  -h, --help              Show help message
```

### Programmatic Evaluation

For custom evaluation workflows:

```python
from evaluate.evaluate_models import evaluate_model, evaluate_all_models
from train.datasets import FilteredCIFAR10, load_natural_dataset

# Option 1: Evaluate a single model
train_images, train_labels, _ = load_natural_dataset("./data/cifar10_natural", "train")
test_images, test_labels, _ = load_natural_dataset("./data/cifar10_natural", "test")

dataset = FilteredCIFAR10(
    data_path="./data/cifar10_natural",
    dataset_type="natural",
    train_images=train_images,
    train_labels=train_labels,
    test_images=test_images,
    test_labels=test_labels,
)

results = evaluate_model(
    checkpoint_path="./models/resnet18_natural/checkpoint.pt",
    dataset=dataset,
    model_name="natural",
    batch_size=128,
    device="cuda:0",
)

print(f"Natural Accuracy: {results['natural_accuracy']:.2f}%")
print(f"Natural Loss: {results['natural_loss']:.4f}")

# Option 2: Evaluate all models
results = evaluate_all_models(
    models_dir="./models",
    natural_dir="./data/cifar10_natural",
    filtered_base_dir="./data",
    output_dir="./evaluation_results",
    log_dir="./logs/evaluation",
    batch_size=128,
    device="cuda:0",
)
```

### Evaluation Module Documentation

#### `evaluate/evaluate_models.py`

Evaluation pipeline script:

- `evaluate_model()`: Evaluate a single trained model on a test dataset
- `evaluate_all_models()`: Evaluate all trained models on multiple test sets
- `run_evaluation_pipeline()`: Convenience wrapper for the full evaluation pipeline
- `cleanup_gpu_memory()`: Clean up GPU memory after evaluation
- `get_evaluation_args()`: Create evaluation arguments for the robustness library
- `setup_logging()`: Set up verbose logging for evaluation

The evaluation pipeline computes:
- **Natural Accuracy**: Standard test accuracy on clean images
- **Natural Loss**: Cross-entropy loss on clean images
- **Adversarial Accuracy**: (optional) Accuracy under adversarial attacks
- **Adversarial Loss**: (optional) Loss under adversarial attacks

### Evaluation Results Format

The evaluation results JSON file contains:

```json
{
  "config": {
    "models_dir": "./models",
    "natural_dir": "./data/cifar10_natural",
    "batch_size": 128,
    ...
  },
  "evaluations": {
    "natural_on_natural": {
      "model_name": "natural_on_natural",
      "checkpoint_path": "models/resnet18_natural/checkpoint.pt",
      "natural_accuracy": 85.23,
      "natural_loss": 0.4523,
      "evaluation_time_seconds": 12.34,
      ...
    },
    "resnet18_high_variance_r5_on_natural": {
      ...
    },
    ...
  },
  "start_time": "2025-11-26T10:00:00",
  "end_time": "2025-11-26T10:05:00"
}
```

## Adversarial Example Generation

The adversarial generation pipeline uses the [robustness library](https://github.com/MadryLab/robustness) to generate adversarial examples for all trained models using PGD and FGSM attacks.

**Key Features:**
- Each model uses its **corresponding dataset** (natural or filtered) for generating adversarial examples
- Adversarial examples are generated for **both train and test sets**
- Attack success rates are reported separately for train and test

### Quick Start: Generate Adversarial Examples

```bash
# Generate adversarial examples for all models (requires --device flag)
bash experiment_scripts/run_adversarial.sh --device cuda:0

# Generate using a specific GPU
bash experiment_scripts/run_adversarial.sh --device 1

# Generate only PGD examples
bash experiment_scripts/run_adversarial.sh --device cuda:0 --attack-types pgd

# Generate only for natural model
bash experiment_scripts/run_adversarial.sh --device cuda:0 --natural-only
```

### Attack Types

The pipeline supports two attack types:

1. **PGD (Projected Gradient Descent)**: A powerful iterative attack that takes multiple gradient steps within an epsilon ball. This is the primary attack method for evaluating adversarial robustness.

2. **FGSM (Fast Gradient Sign Method)**: A single-step attack that is faster but less powerful than PGD. Implemented as PGD with one iteration.

### Attack Configuration

Attack parameters are defined as global variables in `adversarial/generate_adversarial.py` and can be modified:

```python
# PGD Attack Configuration
PGD_CONFIG = {
    'eps': 8.0 / 255.0,          # Maximum perturbation (8/255 is standard for CIFAR-10)
    'step_size': 2.0 / 255.0,    # Step size per iteration
    'iterations': 20,             # Number of attack iterations
    'constraint': 'inf',          # L-infinity norm constraint
    'random_start': True,         # Random initialization within epsilon ball
}

# FGSM Attack Configuration
FGSM_CONFIG = {
    'eps': 8.0 / 255.0,          # Maximum perturbation
    'step_size': 8.0 / 255.0,    # Step size = eps for single-step
    'iterations': 1,              # Single step
    'constraint': 'inf',          # L-infinity norm constraint
    'random_start': False,        # No random start for pure FGSM
}
```

### Adversarial Output

After generation completes, you'll find:

```
adversarial_data/
├── resnet18_natural/
│   ├── pgd/
│   │   ├── train/
│   │   │   ├── adversarial_images.npy   # Train adversarial images (N, H, W, C) uint8
│   │   │   ├── labels.npy               # Train original labels (N,)
│   │   │   ├── clean_images.npy         # Train clean images (N, H, W, C) uint8
│   │   │   └── metadata.json            # Train split stats
│   │   ├── test/
│   │   │   ├── adversarial_images.npy   # Test adversarial images
│   │   │   ├── labels.npy               # Test original labels
│   │   │   ├── clean_images.npy         # Test clean images
│   │   │   └── metadata.json            # Test split stats
│   │   └── metadata.json                # Combined metadata with train/test stats
│   └── fgsm/
│       ├── train/
│       │   └── ...
│       └── test/
│           └── ...
├── resnet18_high_variance_r5/
│   └── ...  # Uses data/filtered_r5/cifar10_{train,test}_high_variance_r5.0/
├── resnet18_high_variance_r10/
│   └── ...  # Uses data/filtered_r10/...
├── resnet18_high_variance_r15/
│   └── ...  # Uses data/filtered_r15/...
├── resnet18_low_variance_r5/
│   └── ...  # Uses data/filtered_r5/cifar10_{train,test}_low_variance_r5.0/
├── resnet18_low_variance_r10/
│   └── ...
├── resnet18_low_variance_r15/
│   └── ...
└── adversarial_generation_results.json  # Summary of all generations

logs/adversarial/
└── adversarial_pgd_YYYYMMDD_HHMMSS.log  # Detailed generation logs
```

**Note:** Each model uses its corresponding dataset:
- `resnet18_natural` → uses `data/cifar10_natural/`
- `resnet18_high_variance_r15` → uses `data/filtered_r15/cifar10_{train,test}_high_variance_r15.0/`
- `resnet18_low_variance_r10` → uses `data/filtered_r10/cifar10_{train,test}_low_variance_r10.0/`

### Command-Line Options

```bash
bash experiment_scripts/run_adversarial.sh --device DEVICE [OPTIONS]

Required:
  --device DEVICE         GPU device to use (e.g., 'cuda:0', 'cuda:1', '0', '1')

Options:
  --models-dir DIR        Directory containing trained models (default: ./models)
  --natural-dir DIR       Path to natural CIFAR-10 dataset (default: ./data/cifar10_natural)
  --output-dir DIR        Directory for adversarial datasets (default: ./adversarial_data)
  --log-dir DIR           Directory for generation logs (default: ./logs/adversarial)
  --batch-size N          Batch size for generation (default: 64)
  --workers N             Number of data loading workers (default: 4)
  --attack-types "T1 T2"  Space-separated attack types (default: "pgd fgsm")
  --cutoff-radii "R1 R2"  Space-separated cutoff radii (default: "5 10 15")
  --natural-only          Only generate for natural model
  --filtered-only         Only generate for filtered models
  --no-natural            Skip natural model
  --no-filtered           Skip filtered models
  -h, --help              Show help message
```

### Programmatic Adversarial Generation

For custom generation workflows:

```python
from adversarial.generate_adversarial import (
    generate_adversarial_dataset,
    run_adversarial_pipeline,
    load_dataset_for_model,
    PGD_CONFIG,
    FGSM_CONFIG,
)
from train.datasets import load_natural_dataset

# Option 1: Generate for a single model (with both train and test)
train_images, train_labels, _ = load_natural_dataset("./data/cifar10_natural", "train")
test_images, test_labels, _ = load_natural_dataset("./data/cifar10_natural", "test")

results = generate_adversarial_dataset(
    model_path="./models/resnet18_natural/checkpoint.pt.best",
    train_images=train_images,
    train_labels=train_labels,
    test_images=test_images,
    test_labels=test_labels,
    output_dir="./adversarial_data/resnet18_natural/pgd",
    attack_type="pgd",
    device="cuda:0",
    batch_size=64,
)

print(f"Train attack success rate: {results['train']['attack_success_rate']:.2f}%")
print(f"Test attack success rate: {results['test']['attack_success_rate']:.2f}%")

# Option 2: Load correct dataset for a model automatically
train_imgs, train_lbls, test_imgs, test_lbls = load_dataset_for_model(
    model_name="resnet18_high_variance_r15",
    filtered_base_dir="./data",
    natural_dir="./data/cifar10_natural",
)

# Option 3: Run the full pipeline (handles all models and datasets automatically)
results = run_adversarial_pipeline(
    models_dir="./models",
    natural_dir="./data/cifar10_natural",
    filtered_base_dir="./data",
    output_dir="./adversarial_data",
    log_dir="./logs/adversarial",
    device="cuda:0",
    attack_types=['pgd', 'fgsm'],
    cutoff_radii=[5, 10, 15],
)
```

### Adversarial Module Documentation

#### `adversarial/generate_adversarial.py`

Adversarial example generation script:

- `generate_adversarial_dataset()`: Generate adversarial examples for both train and test sets
- `generate_adversarial_for_split()`: Generate adversarial examples for a single split (train or test)
- `run_adversarial_pipeline()`: Run the complete generation pipeline for all models
- `load_dataset_for_model()`: Load the correct train/test data for a given model
- `get_dataset_paths()`: Get the correct data paths for a model (natural or filtered)
- `get_attack_config()`: Get attack configuration for a given attack type
- `setup_logging()`: Set up logging for generation
- `cleanup_gpu_memory()`: Clean up GPU memory after generation

### GPU Memory Management

The adversarial generation pipeline:

- Uses the mandatory `--device` flag to ensure all computation stays on a single specified GPU
- Sets `CUDA_VISIBLE_DEVICES` in the shell script to isolate the GPU
- Cleans up GPU memory after processing each model
- This is critical on shared compute clusters where GPU isolation is important

### Loading Adversarial Examples

```python
import numpy as np
import json

# Load adversarial dataset (test set)
adv_dir = "./adversarial_data/resnet18_natural/pgd"
test_adv_images = np.load(f"{adv_dir}/test/adversarial_images.npy")  # (N, H, W, C) uint8
test_labels = np.load(f"{adv_dir}/test/labels.npy")                   # (N,)
test_clean_images = np.load(f"{adv_dir}/test/clean_images.npy")       # (N, H, W, C) uint8

# Load train set adversarial examples
train_adv_images = np.load(f"{adv_dir}/train/adversarial_images.npy")
train_labels = np.load(f"{adv_dir}/train/labels.npy")

# Load combined metadata
with open(f"{adv_dir}/metadata.json") as f:
    metadata = json.load(f)

print(f"Attack type: {metadata['attack_type']}")
print(f"Train attack success rate: {metadata['train']['attack_success_rate']:.2f}%")
print(f"Test attack success rate: {metadata['test']['attack_success_rate']:.2f}%")
print(f"Train clean accuracy: {metadata['train']['clean_accuracy']:.2f}%")
print(f"Test clean accuracy: {metadata['test']['clean_accuracy']:.2f}%")
```

## Workflow Example

### Using Pipeline Script (Simplest Method)

For most use cases, simply use the provided pipeline scripts:

```bash
# Step 1: Create filtered datasets
for r in 5 10 15; do
  python filters/pipeline.py \
    --cutoff-radius $r \
    --filtered-dataset-dir ./data/filtered_r${r}
done

# Step 2: Validate the datasets
bash experiment_scripts/run_validation.sh

# Step 3: Train all models
bash experiment_scripts/run_training.sh --epochs 100

# Step 4: Evaluate all models
bash experiment_scripts/run_evaluation.sh

# Step 5: Generate adversarial examples
bash experiment_scripts/run_adversarial.sh --device cuda:0
```

### Complete Pipeline (Programmatic)

For advanced use cases requiring custom logic:

```python
from filters.utils import load_cifar10, save_cifar10, create_dataloader
from filters.dft import process_dataset, load_filtered_dataset
from train.train_models import train_single_model
from train.datasets import FilteredCIFAR10, load_natural_dataset
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

# 5. Create dataset wrapper for robustness library
train_images, train_labels, _ = load_natural_dataset("./data/cifar10_natural", "train")
test_images, test_labels, _ = load_natural_dataset("./data/cifar10_natural", "test")

dataset = FilteredCIFAR10(
    data_path="./data/cifar10_natural",
    dataset_type="natural",
    train_images=train_images,
    train_labels=train_labels,
    test_images=test_images,
    test_labels=test_labels,
)

# 6. Train model using robustness library
checkpoint_path, stats = train_single_model(
    dataset=dataset,
    output_dir="./models",
    model_name="resnet18_natural",
    epochs=100,
)

# 7. Create adversarial examples for each model
from adversarial.generate_adversarial import generate_adversarial_dataset

# Generate PGD adversarial examples (for both train and test)
results = generate_adversarial_dataset(
    model_path="./models/resnet18_natural/checkpoint.pt.best",
    train_images=train_images,
    train_labels=train_labels,
    test_images=test_images,
    test_labels=test_labels,
    output_dir="./adversarial_data/resnet18_natural/pgd",
    attack_type="pgd",
    device="cuda:0",
)
print(f"PGD train attack success: {results['train']['attack_success_rate']:.2f}%")
print(f"PGD test attack success: {results['test']['attack_success_rate']:.2f}%")

# Generate FGSM adversarial examples
results = generate_adversarial_dataset(
    model_path="./models/resnet18_natural/checkpoint.pt.best",
    train_images=train_images,
    train_labels=train_labels,
    test_images=test_images,
    test_labels=test_labels,
    output_dir="./adversarial_data/resnet18_natural/fgsm",
    attack_type="fgsm",
    device="cuda:0",
)
print(f"FGSM train attack success: {results['train']['attack_success_rate']:.2f}%")
print(f"FGSM test attack success: {results['test']['attack_success_rate']:.2f}%")

# 8. Create ensemble and test adversarial robustness
# TODO: Implement ensemble and adversarial testing
# ensemble = Ensemble([model_high_var, model_low_var])
# test_adversarial_robustness(ensemble, test_loader)
```

## Next Steps

### Immediate Tasks

1. ~~**Run Validation**: Execute `bash experiment_scripts/run_validation.sh` to verify datasets~~ ✓
2. ~~**Model Training**: Implement CNN (ResNet-18) training on filtered datasets~~ ✓
3. ~~**Adversarial Example Generation**: Implement PGD and FGSM attacks using the robustness library~~ ✓
4. **Ensemble Creation**: Implement voting schemes for combining model predictions
5. **Hyperparameter Tuning**: Use validation results to select optimal cutoff radius

### Advanced Tasks

1. **Adversarial Robustness Testing**:
   - ~~Implement PGD (Projected Gradient Descent) attacks using the robustness library~~ ✓
   - ~~Implement FGSM (Fast Gradient Sign Method) attacks~~ ✓
   - Evaluate ensemble robustness vs. single model
   - Compare robustness across different cutoff radii
   - Test if low-variance models require larger epsilon for successful attacks

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