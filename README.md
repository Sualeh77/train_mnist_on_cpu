# MNIST Training Pipeline

A PyTorch-based training pipeline for the MNIST dataset, featuring a modular architecture and comprehensive training visualization.

## Project Structure

```
train_mnist_on_cpu/
├── pyproject.toml          # Poetry dependency management
├── README.md              # Project documentation
├── train_mnist_on_cpu/    # Main package directory
│   ├── __init__.py
│   ├── config.py          # Configuration and hyperparameters
│   ├── main.py           # Main training orchestration
│   ├── models.py         # Neural network model definitions
│   ├── mnist_dataset.py  # Dataset and data loading utilities
│   ├── train.py          # Training loop implementation
│   ├── utils.py          # Utility functions
│   └── visualize.py      # Training visualization utilities
├── models/               # Saved model checkpoints
├── plots/               # Training visualization plots
└── datasets/           # MNIST dataset storage
```

## Setup Instructions

1. Create and activate conda environment:
```bash
conda create -n emlo_ass2 python=3.10
conda activate emlo_ass2
```

2. Install Poetry:
```bash
python -m pip install poetry
```
Dont forget to use -m while installing libraries to do env level installations as your pip might still be pointing to system level lib. You can verify it using comands:
```bash
which python
which pip
```

3. Install project dependencies:
```bash
poetry install
python -m pip install torch torchvision torchaudio
```

## Codebase Documentation

### 1. Configuration (`config.py`)
Central configuration file containing all hyperparameters and settings:
- Training hyperparameters (batch size, epochs, learning rate)
- Model configuration (model type, optimizer, scheduler, loss function)
- Device settings (CPU/CUDA/MPS)
- File paths for data, model checkpoints, and plots

### 2. Main Training Orchestration (`main.py`)
Contains the `MNISTTrainer` class that orchestrates the entire training pipeline:

#### `MNISTTrainer` Class
- **Purpose**: Manages the complete training workflow
- **Key Methods**:
  - `__init__`: Initializes training components and directories
  - `_init_components`: Sets up model, loss function, optimizer, and data loaders
  - `train`: Executes the training loop
  - `save_model`: Saves model checkpoints
  - `save_plots`: Generates and saves training visualizations
  - `run`: Orchestrates the complete training pipeline

### 3. Model Definition (`models.py`)
Contains the neural network architecture:

#### `MNISTModelFCN` Class
- **Purpose**: Implements a fully connected neural network for MNIST
- **Architecture**:
  - Input: 784 (28x28 flattened images)
  - Hidden layers: 128 → 256 → 512 → 128
  - Output: 10 (digit classes)
  - Activation: ReLU
- **Functions**:
  - `get_model`: Factory function to create model instances

### 4. Dataset Management (`mnist_dataset.py`)
Handles data loading and preprocessing:

#### `MNISTDatasetFCN` Class
- **Purpose**: Custom Dataset class for MNIST
- **Features**:
  - Image normalization
  - Flattening of 28x28 images
  - Device placement (CPU/CUDA)
- **Functions**:
  - `get_dataloader`: Creates DataLoader instances for training/validation

### 5. Training Implementation (`train.py`)
Implements the training loop:

#### `ModelTrainer` Class
- **Purpose**: Manages the training and validation process
- **Key Methods**:
  - `train_batch`: Handles single batch training
  - `validate_batch`: Handles single batch validation
  - `train`: Implements the complete training loop with metrics tracking

### 6. Utility Functions (`utils.py`)
Provides helper functions for training setup:
- `get_loss_fn`: Creates loss function instances
- `get_optimizer`: Creates optimizer instances
- `get_scheduler`: Creates learning rate scheduler instances

### 7. Visualization (`visualize.py`)
Handles training metrics visualization:

#### Functions
- `plot_batch_metrics`: Creates dual-axis plots for batch-wise metrics
- `plot_epoch_metrics`: Creates dual-axis plots for epoch-wise metrics
- Features:
  - Dual y-axes for loss and accuracy
  - Color-coded training and validation metrics
  - Grid lines and legends
  - High-resolution output

## Usage

1. Configure training parameters in `config.py`

2. Run training:
```bash
python -m train_mnist_on_cpu.main
```

3. Monitor training:
- Progress bars show batch-wise training
- Console output shows epoch-wise metrics
- Plots are saved in the `plots/` directory
- Model checkpoints are saved in the `models/` directory

## Output

The training pipeline generates:
1. Model checkpoints in `models/` directory
2. Training visualizations in `plots/` directory:
   - `batch_metrics.png`: Batch-wise training metrics
   - `epoch_metrics.png`: Epoch-wise training metrics

## Dependencies

- Python 3.10+
- PyTorch
- torchvision
- matplotlib
- tqdm
- pandas
- torch-summary

## Future Improvements

1. Add support for:
   - More model architectures
   - Additional optimizers and schedulers
   - Early stopping
   - Model checkpointing
   - Learning rate scheduling
   - Custom metrics tracking
   - Training resume capability

2. Implement:
   - Docker containerization
   - AWS deployment
   - Model inference pipeline
   - Unit tests
   - CI/CD pipeline
