from mnist_dataset import get_dataloader
from models import get_model
from config import (
    DEVICE,
    SAVE_PATH,
    LOAD_PATH,
    MODEL_TYPE,
    OPTIMIZER_TYPE,
    SCHEDULER_TYPE,
    LOSS_TYPE,
    BATCH_SIZE,
    NUM_EPOCHS,
    LEARNING_RATE,
    CHECKPOINT_DIR,
    BEST_MODEL_DIR,
)
from train import ModelTrainer
from visualize import plot_batch_metrics, plot_epoch_metrics
from utils import get_loss_fn, get_optimizer, get_scheduler
import torch
import os
import argparse


def parse_args():
    parser = argparse.ArgumentParser(description="Train MNIST model")
    parser.add_argument(
        "--resume", action="store_true", help="Resume training from checkpoint"
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Path to checkpoint file to resume from",
    )
    return parser.parse_args()


class MNISTTrainer:
    def __init__(self, resume_training=False, checkpoint_path=None):
        self.device = DEVICE
        self.model_type = MODEL_TYPE
        self.optimizer_type = OPTIMIZER_TYPE
        self.scheduler_type = SCHEDULER_TYPE
        self.loss_type = LOSS_TYPE
        self.batch_size = BATCH_SIZE
        self.num_epochs = NUM_EPOCHS
        self.learning_rate = LEARNING_RATE
        self.resume_training = resume_training
        self.checkpoint_path = checkpoint_path

        self.batch_train_losses = []
        self.batch_val_losses = []
        self.batch_train_accuracies = []
        self.batch_val_accuracies = []
        self.epoch_train_losses = []
        self.epoch_val_losses = []
        self.epoch_train_accuracies = []
        self.epoch_val_accuracies = []

        # Create plots directory
        os.makedirs("../plots/", exist_ok=True)

        # Initialize components
        self._init_components()

    def _init_components(self):
        """Initialize model, loss function, optimizer, and scheduler"""
        self.model = get_model(self.model_type).to(self.device)
        self.loss_fn = get_loss_fn(self.loss_type)
        self.optimizer = get_optimizer(
            self.model, self.optimizer_type, self.learning_rate
        )
        self.scheduler = get_scheduler(
            self.optimizer, self.scheduler_type, self.num_epochs
        )
        self.trainer = ModelTrainer(
            self.model, self.optimizer, self.loss_fn, self.scheduler
        )

        # Initialize data loaders
        self.train_loader = get_dataloader("train", self.batch_size)
        self.val_loader = get_dataloader("test", self.batch_size)

    def train(self):
        """Main training loop"""
        print(f"Starting training on {self.device}...")
        print(f"Model type: {self.model_type}")
        print(f"Optimizer: {self.optimizer_type}")
        print(f"Learning rate: {self.learning_rate}")
        print(f"Batch size: {self.batch_size}")
        print(f"Number of epochs: {self.num_epochs}")
        if self.resume_training and self.checkpoint_path:
            print(f"Resuming training from checkpoint: {self.checkpoint_path}")

        metrics = self.trainer.train(
            num_epochs=self.num_epochs,
            train_loader=self.train_loader,
            val_loader=self.val_loader,
            model_name=self.model_type,
            optimizer_name=self.optimizer_type,
            scheduler_name=self.scheduler_type,
            loss_name=self.loss_type,
            resume_training=self.resume_training,
            checkpoint_path=self.checkpoint_path,
        )

        self.batch_train_losses = metrics[0]
        self.batch_val_losses = metrics[1]
        self.batch_train_accuracies = metrics[2]
        self.batch_val_accuracies = metrics[3]
        self.epoch_train_losses = metrics[4]
        self.epoch_val_losses = metrics[5]
        self.epoch_train_accuracies = metrics[6]
        self.epoch_val_accuracies = metrics[7]

    def save_plots(self):
        """Generate and save training plots"""
        plot_batch_metrics(
            self.batch_train_losses,
            self.batch_val_losses,
            self.batch_train_accuracies,
            self.batch_val_accuracies,
            save_path="../plots/batch_metrics.png",
        )

        plot_epoch_metrics(
            self.epoch_train_losses,
            self.epoch_val_losses,
            self.epoch_train_accuracies,
            self.epoch_val_accuracies,
            save_path="../plots/epoch_metrics.png",
        )
        print("Training plots saved to plots/ directory")

    def run(self):
        """Run the complete training pipeline"""
        self.train()
        self.save_plots()


def main():
    args = parse_args()

    # Validate checkpoint path if resume is enabled
    if args.resume and not args.checkpoint:
        raise ValueError("Checkpoint path must be provided when resuming training")

    trainer = MNISTTrainer(resume_training=args.resume, checkpoint_path=args.checkpoint)
    trainer.run()


if __name__ == "__main__":
    main()
