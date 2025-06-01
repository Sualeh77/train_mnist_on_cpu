import torch
from tqdm import tqdm

class ModelTrainer:
    def __init__(self, model, optimizer, loss_fn, scheduler=None):
        self.model = model
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.scheduler = scheduler
        
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
    
    def train(self, num_epochs, train_loader, val_loader):
        batch_train_losses = []
        batch_val_losses = []
        batch_train_accuracies = []
        batch_val_accuracies = []
        epoch_train_losses = []
        epoch_val_losses = []
        epoch_train_accuracies = []
        epoch_val_accuracies = []
        
        for epoch in range(num_epochs):
            print(f"Epoch {epoch+1}/{num_epochs}")

            for images, labels in tqdm(train_loader, desc="Training"):
                batch_train_loss, batch_train_accuracy = self.train_batch(images, labels)
                batch_train_losses.append(batch_train_loss)
                batch_train_accuracies.append(batch_train_accuracy)

            for images, labels in tqdm(val_loader, desc="Validating"):
                batch_val_loss, batch_val_accuracy = self.validate_batch(images, labels)
                batch_val_losses.append(batch_val_loss)
                batch_val_accuracies.append(batch_val_accuracy)

            epoch_train_losses.append(sum(batch_train_losses) / len(batch_train_losses))
            epoch_val_losses.append(sum(batch_val_losses) / len(batch_val_losses))
            epoch_train_accuracies.append(sum(batch_train_accuracies) / len(batch_train_accuracies))
            epoch_val_accuracies.append(sum(batch_val_accuracies) / len(batch_val_accuracies))

            print(f"Epoch {epoch+1:3d}/{num_epochs} | "
                  f"Train Loss: {epoch_train_losses[-1]:.4f} | "
                  f"Train Acc: {epoch_train_accuracies[-1]:.4f} | "
                  f"Val Loss: {epoch_val_losses[-1]:.4f} | "
                  f"Val Acc: {epoch_val_accuracies[-1]:.4f}")

        return batch_train_losses, batch_val_losses, batch_train_accuracies, batch_val_accuracies, \
            epoch_train_losses, epoch_val_losses, epoch_train_accuracies, epoch_val_accuracies