import torch
import torch.nn as nn


class MNISTModelFCN(nn.Module):
    def __init__(self):
        super(MNISTModelFCN, self).__init__()
        self.fc1 = nn.Linear(28 * 28, 128)
        self.fc2 = nn.Linear(128, 256)
        self.fc3 = nn.Linear(256, 512)
        self.fc4 = nn.Linear(512, 128)
        self.fc5 = nn.Linear(128, 10)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.relu(self.fc3(x))
        x = torch.relu(self.fc4(x))
        x = self.fc5(x)
        return x


def get_model(model_type="FCN"):
    if model_type == "FCN":
        return MNISTModelFCN()
    else:
        raise ValueError(f"Model type {model_type} not supported")
