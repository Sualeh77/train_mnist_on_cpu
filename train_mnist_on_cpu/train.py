import torch
from tqdm import tqdm
import os
import glob
from config import (CHECKPOINT_DIR, CHECKPOINT_INTERVAL, 
                   BEST_MODEL_DIR)

class ModelTrainer:
    def __init__(self, model, optimizer, loss_fn, scheduler=None):
        self.model = model
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.scheduler = scheduler
        self.best_val_accuracy = 0.0
        self.best_epoch = 0
        self.best_model_path = None
        
        # Create directories
        os.makedirs(CHECKPOINT_DIR, exist_ok=True)
        os.makedirs(BEST_MODEL_DIR, exist_ok=True)
        
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
            'loss_name': loss_name,
            'best_val_accuracy': self.best_val_accuracy,
            'best_epoch': self.best_epoch,
            'best_model_path': self.best_model_path
        }
        
        checkpoint_path = os.path.join(
            CHECKPOINT_DIR,
            f"checkpoint_{model_name}_{optimizer_name}_{scheduler_name or 'no_scheduler'}_{loss_name}_epoch_{epoch}.pt"
        )
        torch.save(checkpoint, checkpoint_path)
        print(f"\nCheckpoint saved: {checkpoint_path}")
        
    def save_best_model(self, epoch, val_accuracy, model_name, optimizer_name, scheduler_name, loss_name):
        """Save the best model based on validation accuracy and delete previous best model"""
        if val_accuracy > self.best_val_accuracy:
            # Delete previous best model if it exists
            if self.best_model_path and os.path.exists(self.best_model_path):
                try:
                    os.remove(self.best_model_path)
                    print(f"Deleted previous best model: {self.best_model_path}")
                except Exception as e:
                    print(f"Warning: Could not delete previous best model: {e}")
            
            self.best_val_accuracy = val_accuracy
            self.best_epoch = epoch
            
            # Save new best model
            self.best_model_path = os.path.join(
                BEST_MODEL_DIR,
                f"best_model_{model_name}_{optimizer_name}_{scheduler_name or 'no_scheduler'}_{loss_name}_epoch_{epoch}_val_acc_{val_accuracy:.4f}.pt"
            )
            torch.save(self.model.state_dict(), self.best_model_path)
            print(f"\nNew best model saved! Validation accuracy: {val_accuracy:.4f}")
            print(f"Model saved to: {self.best_model_path}")
        
    def load_checkpoint(self, checkpoint_path):
        """Load a checkpoint and restore model and training state"""
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"Checkpoint not found at {checkpoint_path}")
            
        checkpoint = torch.load(checkpoint_path)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        if self.scheduler and checkpoint['scheduler_state_dict']:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        # Restore best model tracking
        self.best_val_accuracy = checkpoint.get('best_val_accuracy', 0.0)
        self.best_epoch = checkpoint.get('best_epoch', 0)
        self.best_model_path = checkpoint.get('best_model_path', None)
            
        print(f"\nLoaded checkpoint from epoch {checkpoint['epoch']}")
        print(f"Best validation accuracy so far: {self.best_val_accuracy:.4f} (epoch {self.best_epoch})")
        if self.best_model_path:
            print(f"Best model path: {self.best_model_path}")
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
    
    def train(self, num_epochs, train_loader, val_loader, model_name, optimizer_name, scheduler_name, loss_name, start_epoch=0, resume_training=False, checkpoint_path=None):
        batch_train_losses = []
        batch_val_losses = []
        batch_train_accuracies = []
        batch_val_accuracies = []
        epoch_train_losses = []
        epoch_val_losses = []
        epoch_train_accuracies = []
        epoch_val_accuracies = []
        
        # Load checkpoint if resuming training
        if resume_training and checkpoint_path:
            start_epoch, metrics = self.load_checkpoint(checkpoint_path)
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
            
            # Save best model if validation accuracy improved
            self.save_best_model(
                epoch + 1,
                epoch_val_accuracies[-1],
                model_name,
                optimizer_name,
                scheduler_name,
                loss_name
            )
            
            # Save checkpoint at specified intervals
            if (epoch + 1) % CHECKPOINT_INTERVAL == 0:
                metrics = (batch_train_losses, batch_val_losses, 
                          batch_train_accuracies, batch_val_accuracies,
                          epoch_train_losses, epoch_val_losses,
                          epoch_train_accuracies, epoch_val_accuracies)
                self.save_checkpoint(epoch + 1, metrics, model_name, optimizer_name, scheduler_name, loss_name)

        print(f"\nTraining completed. Best validation accuracy: {self.best_val_accuracy:.4f} (epoch {self.best_epoch})")
        return batch_train_losses, batch_val_losses, batch_train_accuracies, batch_val_accuracies, \
               epoch_train_losses, epoch_val_losses, epoch_train_accuracies, epoch_val_accuracies