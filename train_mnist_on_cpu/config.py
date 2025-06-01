# Hyperparameters

BATCH_SIZE = 32
NUM_EPOCHS = 10
LEARNING_RATE = 1e-2

# Model
MODEL_TYPE = "FCN"

# Optimizer
OPTIMIZER_TYPE = "SGD"

# Scheduler
SCHEDULER_TYPE = None

# Loss
LOSS_TYPE = "cross_entropy"

# Device
import torch
DEVICE = "cpu" #"cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"

# Save path
SAVE_PATH = "../models/"

# Load path
LOAD_PATH = None

# Data
DATA_DIR = "../datasets/mnist"