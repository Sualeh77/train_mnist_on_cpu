from torch.utils.data import Dataset, DataLoader
from torchvision import datasets
import torch
import torch.nn as nn

from config import DEVICE, DATA_DIR

train_data = datasets.MNIST(DATA_DIR, train=True, download=True)
train_images = train_data.data
train_labels = train_data.targets

test_data = datasets.MNIST(DATA_DIR, train=False, download=True)
test_images = test_data.data
test_labels = test_data.targets

class MNISTDatasetFCN(Dataset):
    def __init__(self, images, labels):
        images = images.float()
        images = images.view(-1, 28 * 28)
        self.x = images
        self.y = labels

    def __len__(self):
        return len(self.x)
    
    def __getitem__(self, idx):
        x, y = self.x[idx], self.y[idx]
        return x.to(DEVICE), y.to(DEVICE)
    
def get_dataloader(train_test="train", batch_size=32):
    if train_test == "train":
        dataset = MNISTDatasetFCN(train_images, train_labels)
    else:
        dataset = MNISTDatasetFCN(test_images, test_labels)
    return DataLoader(dataset, batch_size=batch_size, shuffle=True, pin_memory=True)
