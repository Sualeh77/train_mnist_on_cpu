import torch
import torch.nn as nn

def get_loss_fn(loss_type="cross_entropy"):
    if loss_type == "cross_entropy":
        return nn.CrossEntropyLoss()
    else:
        raise ValueError(f"Loss type {loss_type} not supported")
    
def get_optimizer(model, optimizer_type="SGD", learning_rate=1e-2):
    if optimizer_type == "SGD":
        return torch.optim.SGD(model.parameters(), lr=learning_rate)
    else:
        raise ValueError(f"Optimizer type {optimizer_type} not supported")
    
def get_scheduler(optimizer, scheduler_type=None, num_epochs=100, scheduler_kwargs={}):
    if scheduler_type is None:
        return None
    else:
        raise ValueError(f"Scheduler type {scheduler_type} not supported")