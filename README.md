# MNIST Training Pipeline

A PyTorch-based training pipeline for the MNIST dataset, featuring a modular architecture, comprehensive training visualization, and Docker support for easy deployment.

## Project Structure

```
train_mnist_on_cpu/
├── pyproject.toml          # Poetry dependency management
├── README.md              # Project documentation
├── Dockerfile             # Docker configuration
├── entrypoint.sh          # Docker entrypoint script
├── format_code.sh         # Code formatting script
├── .dockerignore         # Docker ignore file
├── train_mnist_on_cpu/    # Main package directory
│   ├── __init__.py
│   ├── config.py          # Configuration and hyperparameters
│   ├── main.py           # Main training orchestration
│   ├── models.py         # Neural network model definitions
│   ├── mnist_dataset.py  # Dataset and data loading utilities
│   ├── train.py          # Training loop implementation
│   ├── utils.py          # Utility functions
│   └── visualize.py      # Training visualization utilities
├── models/               # Saved model checkpoints and best models
│   ├── checkpoints/     # Training checkpoints
│   └── best_model/      # Best model based on validation accuracy
├── plots/               # Training visualization plots
└── datasets/           # MNIST dataset storage
```

## Setup Instructions

### Option 1: Local Setup

1. Create and activate conda environment:
```bash
conda create -n emlo_ass2 python=3.10
conda activate emlo_ass2
```

2. Install Poetry:
```bash
python -m pip install poetry
```
Note: Always use `python -m pip` to ensure environment-level installations. Verify your Python and pip paths:
```bash
which python
which pip
```

3. Install project dependencies:
```bash
# Install project dependencies using Poetry
poetry install

# Install PyTorch (CPU version)
python -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
```

### Option 2: Docker Setup

1. Build the Docker image:
```bash
# From the project root directory
docker build -t mnist-trainer .
```

2. Create required directories (if they don't exist):
```bash
mkdir -p datasets models/checkpoints models/best_model plots
```

## Running the Training

### Option 1: Local Training

1. Configure training parameters in `config.py`:
   - Adjust hyperparameters (batch size, epochs, learning rate)
   - Select model type, optimizer, and scheduler
   - Modify paths if needed

2. Start training from scratch:
```bash
python -m train_mnist_on_cpu.main
```

3. Resume training from a checkpoint:
```bash
python -m train_mnist_on_cpu.main --resume --checkpoint models/checkpoints/checkpoint_FCN_SGD_no_scheduler_cross_entropy_epoch_X.pt
```

### Option 2: Docker Training

1. Training from scratch:
```bash
docker run -ti --rm \
    -v $(pwd)/datasets:/app/datasets \
    -v $(pwd)/models:/app/models \
    -v $(pwd)/plots:/app/plots \
    mnist-trainer
```

2. Resume training from a checkpoint:
```bash
docker run -ti --rm \
    -v $(pwd)/datasets:/app/datasets \
    -v $(pwd)/models:/app/models \
    -v $(pwd)/plots:/app/plots \
    mnist-trainer --resume --checkpoint /app/models/checkpoints/checkpoint_FCN_SGD_no_scheduler_cross_entropy_epoch_X.pt
```

## Training Features

### 1. Model Checkpointing
- Automatic checkpoint saving every N epochs (configurable in `config.py`)
- Checkpoints include:
  - Model state
  - Optimizer state
  - Scheduler state
  - Training metrics
  - Best model information

### 2. Best Model Tracking
- Saves the model with best validation accuracy
- Automatically deletes previous best model
- Best model filename includes:
  - Model type
  - Optimizer type
  - Scheduler type
  - Loss type
  - Epoch number
  - Validation accuracy

### 3. Training Visualization
- Real-time progress bars for batch-wise training
- Console output for epoch-wise metrics
- Dual-axis plots for:
  - Batch-wise metrics (`plots/batch_metrics.png`)
  - Epoch-wise metrics (`plots/epoch_metrics.png`)
- Features:
  - Separate axes for loss and accuracy
  - Color-coded training and validation metrics
  - Grid lines and legends
  - High-resolution output

## Codebase Documentation

### 1. Configuration (`config.py`)
Central configuration file containing:
- Training hyperparameters:
  - `BATCH_SIZE`: Number of samples per batch
  - `NUM_EPOCHS`: Total training epochs
  - `LEARNING_RATE`: Initial learning rate
- Model settings:
  - `MODEL_TYPE`: Neural network architecture
  - `OPTIMIZER_TYPE`: Optimization algorithm
  - `SCHEDULER_TYPE`: Learning rate scheduler
  - `LOSS_TYPE`: Loss function
- Path configurations:
  - `CHECKPOINT_DIR`: Checkpoint storage
  - `BEST_MODEL_DIR`: Best model storage
  - `CHECKPOINT_INTERVAL`: Epochs between checkpoints

### 2. Main Training Orchestration (`main.py`)
`MNISTTrainer` class features:
- Command-line argument parsing for training control
- Training component initialization
- Complete training pipeline management
- Plot generation and saving

### 3. Training Implementation (`train.py`)
`ModelTrainer` class capabilities:
- Training and validation loops
- Checkpoint management
- Best model tracking
- Metrics collection
- Training resumption

### 4. Model Architecture (`models.py`)
- Fully Connected Network (FCN) for MNIST
- Architecture:
  - Input: 784 (28x28 flattened images)
  - Hidden layers: 128 → 256 → 512 → 128
  - Output: 10 (digit classes)
  - Activation: ReLU

### 5. Data Management (`mnist_dataset.py`)
- MNIST dataset handling
- Data normalization
- DataLoader creation
- Device placement

## Output Structure

### Models Directory
```
models/
├── checkpoints/
│   └── checkpoint_[model]_[optimizer]_[scheduler]_[loss]_epoch_[N].pt
└── best_model/
    └── best_model_[model]_[optimizer]_[scheduler]_[loss]_epoch_[N]_val_acc_[X].pt
```

### Plots Directory
```
plots/
├── batch_metrics.png  # Batch-wise training visualization
└── epoch_metrics.png  # Epoch-wise training visualization
```

## Dependencies

- Python 3.10+
- PyTorch (CPU version)
- torchvision
- matplotlib
- tqdm
- pandas
- torch-summary
- Poetry (for dependency management)

## Future Improvements

1. Model Enhancements:
   - Additional architectures (CNN, RNN)
   - More optimizers and schedulers
   - Early stopping
   - Learning rate finder
   - Custom metrics

2. Infrastructure:
   - Multi-GPU support
   - Distributed training
   - Model serving
   - CI/CD pipeline
   - Unit tests
   - AWS deployment

3. Features:
   - Experiment tracking
   - Hyperparameter tuning
   - Model quantization
   - ONNX export
   - Inference pipeline

## Development Setup

### Code Formatting

This project uses Black for code formatting. To set up and use Black:

1. Install development dependencies:
```bash
poetry install --with dev
```

2. Format all Python files:
```bash
./format_code.sh
```

Or format specific files:
```bash
poetry run black path/to/file.py
```

Black configuration (in `pyproject.toml`):
- Line length: 88 characters
- Target Python version: 3.10
- Excludes: git, build, and virtual environment directories
