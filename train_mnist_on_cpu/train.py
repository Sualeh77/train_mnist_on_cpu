import torch
from tqdm import tqdm
import os
from config import CHECKPOINT_DIR, CHECKPOINT_INTERVAL, RESUME_TRAINING, CHECKPOINT_PATH

class ModelTrainer:
    def __init__(self, model, optimizer, loss_fn, scheduler=None):
        self.model = model
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.scheduler = scheduler
        
        # Create checkpoint directory
        os.makedirs(CHECKPOINT_DIR, exist_ok=True)
        
    def save_checkpoint(self, epoch, metrics, model_name, optimizer_name, scheduler_name, loss_name):
        """Save a checkpoint of the model and training state"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict() if self.scheduler else None,
            'metrics': metrics,
            'model_name': model_name,
            'optimizer_name': optimizer_name,
            'scheduler_name': scheduler_name,
            'loss_name': loss_name
        }
        
        checkpoint_path = os.path.join(
            CHECKPOINT_DIR,
            f"checkpoint_{model_name}_{optimizer_name}_{scheduler_name or 'no_scheduler'}_{loss_name}_epoch_{epoch}.pt"
        )
        torch.save(checkpoint, checkpoint_path)
        print(f"\nCheckpoint saved: {checkpoint_path}")
        
    def load_checkpoint(self, checkpoint_path):
        """Load a checkpoint and restore model and training state"""
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"Checkpoint not found at {checkpoint_path}")
            
        checkpoint = torch.load(checkpoint_path)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        if self.scheduler and checkpoint['scheduler_state_dict']:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            
        print(f"\nLoaded checkpoint from epoch {checkpoint['epoch']}")
        return checkpoint['epoch'], checkpoint['metrics']
        
    def train_batch(self, images, labels):
        self.model.train()
        predictions = self.model(images)
        batch_loss = self.loss_fn(predictions, labels)
        batch_loss.backward()
        self.optimizer.step()
        self.optimizer.zero_grad()

        correct_predictions = predictions.argmax(dim=1) == labels
        batch_accuracy = correct_predictions.float().mean()
        return batch_loss.item(), batch_accuracy.item()

    def validate_batch(self, images, labels):
        self.model.eval()
        with torch.no_grad():
            predictions = self.model(images)
            batch_loss = self.loss_fn(predictions, labels)
            correct_predictions = predictions.argmax(dim=1) == labels
            batch_accuracy = correct_predictions.float().mean()
            return batch_loss.item(), batch_accuracy.item()
    
    def train(self, num_epochs, train_loader, val_loader, model_name, optimizer_name, scheduler_name, loss_name, start_epoch=0):
        batch_train_losses = []
        batch_val_losses = []
        batch_train_accuracies = []
        batch_val_accuracies = []
        epoch_train_losses = []
        epoch_val_losses = []
        epoch_train_accuracies = []
        epoch_val_accuracies = []
        
        # Load checkpoint if resuming training
        if RESUME_TRAINING and CHECKPOINT_PATH:
            start_epoch, metrics = self.load_checkpoint(CHECKPOINT_PATH)
            batch_train_losses = metrics[0]
            batch_val_losses = metrics[1]
            batch_train_accuracies = metrics[2]
            batch_val_accuracies = metrics[3]
            epoch_train_losses = metrics[4]
            epoch_val_losses = metrics[5]
            epoch_train_accuracies = metrics[6]
            epoch_val_accuracies = metrics[7]
        
        for epoch in range(start_epoch, num_epochs):
            print(f"Epoch {epoch+1}/{num_epochs}")

            for images, labels in tqdm(train_loader, desc="Training"):
                batch_train_loss, batch_train_accuracy = self.train_batch(images, labels)
                batch_train_losses.append(batch_train_loss)
                batch_train_accuracies.append(batch_train_accuracy)

            for images, labels in tqdm(val_loader, desc="Validating"):
                batch_val_loss, batch_val_accuracy = self.validate_batch(images, labels)
                batch_val_losses.append(batch_val_loss)
                batch_val_accuracies.append(batch_val_accuracy)

            epoch_train_losses.append(sum(batch_train_losses[-len(train_loader):]) / len(train_loader))
            epoch_val_losses.append(sum(batch_val_losses[-len(val_loader):]) / len(val_loader))
            epoch_train_accuracies.append(sum(batch_train_accuracies[-len(train_loader):]) / len(train_loader))
            epoch_val_accuracies.append(sum(batch_val_accuracies[-len(val_loader):]) / len(val_loader))

            print(f"Epoch {epoch+1:3d}/{num_epochs} | "
                  f"Train Loss: {epoch_train_losses[-1]:.4f} | "
                  f"Train Acc: {epoch_train_accuracies[-1]:.4f} | "
                  f"Val Loss: {epoch_val_losses[-1]:.4f} | "
                  f"Val Acc: {epoch_val_accuracies[-1]:.4f}")
            
            # Save checkpoint at specified intervals
            if (epoch + 1) % CHECKPOINT_INTERVAL == 0:
                metrics = (batch_train_losses, batch_val_losses, 
                          batch_train_accuracies, batch_val_accuracies,
                          epoch_train_losses, epoch_val_losses,
                          epoch_train_accuracies, epoch_val_accuracies)
                self.save_checkpoint(epoch + 1, metrics, model_name, optimizer_name, scheduler_name, loss_name)

        return batch_train_losses, batch_val_losses, batch_train_accuracies, batch_val_accuracies, \
               epoch_train_losses, epoch_val_losses, epoch_train_accuracies, epoch_val_accuracies