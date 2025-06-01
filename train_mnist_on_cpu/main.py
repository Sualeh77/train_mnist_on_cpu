from mnist_dataset import get_dataloader
from models import get_model
from config import DEVICE, SAVE_PATH, LOAD_PATH, MODEL_TYPE, OPTIMIZER_TYPE, SCHEDULER_TYPE, LOSS_TYPE, BATCH_SIZE, NUM_EPOCHS, LEARNING_RATE
from train import ModelTrainer
from visualize import plot_batch_metrics, plot_epoch_metrics
from utils import get_loss_fn, get_optimizer, get_scheduler
import torch
import os

class MNISTTrainer:
    def __init__(self):
        self.device = DEVICE
        self.model_type = MODEL_TYPE
        self.optimizer_type = OPTIMIZER_TYPE
        self.scheduler_type = SCHEDULER_TYPE
        self.loss_type = LOSS_TYPE
        self.batch_size = BATCH_SIZE
        self.num_epochs = NUM_EPOCHS
        self.learning_rate = LEARNING_RATE
        self.save_path = SAVE_PATH

        self.batch_train_losses = []
        self.batch_val_losses = []
        self.batch_train_accuracies = []
        self.batch_val_accuracies = []
        self.epoch_train_losses = []
        self.epoch_val_losses = []
        self.epoch_train_accuracies = []
        self.epoch_val_accuracies = []
        
        # Create save directory
        os.makedirs(os.path.dirname(self.save_path), exist_ok=True)
        os.makedirs("../plots/", exist_ok=True)
        
        # Initialize components
        self._init_components()
    
    def _init_components(self):
        """Initialize model, loss function, optimizer, and scheduler"""
        self.model = get_model(self.model_type).to(self.device)
        self.loss_fn = get_loss_fn(self.loss_type)
        self.optimizer = get_optimizer(self.model, self.optimizer_type, self.learning_rate)
        self.scheduler = get_scheduler(self.optimizer, self.scheduler_type, self.num_epochs)
        self.trainer = ModelTrainer(self.model, self.optimizer, self.loss_fn, self.scheduler)
        
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
        
        metrics = self.trainer.train(self.num_epochs, self.train_loader, self.val_loader)

        self.batch_train_losses = metrics[0]
        self.batch_val_losses = metrics[1]
        self.batch_train_accuracies = metrics[2]
        self.batch_val_accuracies = metrics[3]
        self.epoch_train_losses = metrics[4]
        self.epoch_val_losses = metrics[5]
        self.epoch_train_accuracies = metrics[6]
        self.epoch_val_accuracies = metrics[7]
        print(self.epoch_train_accuracies)
    
    def save_model(self):
        """Save the trained model"""
        torch.save(self.model.state_dict(), self.save_path + f"{self.model_type}_{self.optimizer_type}_{'' if not self.scheduler_type else self.scheduler_type}_{self.loss_type}.pth")
        print(f"\nModel saved to {self.save_path}")
    
    def save_plots(self):
        """Generate and save training plots"""
        plot_batch_metrics(
            self.batch_train_losses,
            self.batch_val_losses,
            self.batch_train_accuracies,
            self.batch_val_accuracies,
            save_path="../plots/batch_metrics.png"
        )
        
        plot_epoch_metrics(
            self.epoch_train_losses,
            self.epoch_val_losses,
            self.epoch_train_accuracies,
            self.epoch_val_accuracies,
            save_path="../plots/epoch_metrics.png"
        )
        print("Training plots saved to plots/ directory")
    
    def run(self):
        """Run the complete training pipeline"""
        self.train()
        self.save_model()
        self.save_plots()

def main():
    trainer = MNISTTrainer()
    trainer.run()

if __name__ == "__main__":
    main()
