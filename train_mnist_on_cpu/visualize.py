import matplotlib.pyplot as plt
import numpy as np


def plot_batch_metrics(
    batch_train_losses,
    batch_val_losses,
    batch_train_accuracies,
    batch_val_accuracies,
    save_path=None,
):
    """
    Plot batch-wise training and validation losses and accuracies using dual axes.

    Args:
        batch_train_losses (list): List of training losses per batch
        batch_val_losses (list): List of validation losses per batch
        batch_train_accuracies (list): List of training accuracies per batch
        batch_val_accuracies (list): List of validation accuracies per batch
        save_path (str, optional): Path to save the plot. If None, plot is displayed.
    """
    fig, ax1 = plt.subplots(figsize=(12, 6))

    # Create second y-axis
    ax2 = ax1.twinx()

    # Plot losses on primary y-axis
    line1 = ax1.plot(batch_train_losses, label="Training Loss", color="blue", alpha=0.7)
    line2 = ax1.plot(batch_val_losses, label="Validation Loss", color="red", alpha=0.7)
    ax1.set_xlabel("Batch")
    ax1.set_ylabel("Loss", color="black")
    ax1.tick_params(axis="y", labelcolor="black")

    # Plot accuracies on secondary y-axis
    line3 = ax2.plot(
        batch_train_accuracies, label="Training Accuracy", color="green", alpha=0.7
    )
    line4 = ax2.plot(
        batch_val_accuracies, label="Validation Accuracy", color="purple", alpha=0.7
    )
    ax2.set_ylabel("Accuracy", color="black")
    ax2.tick_params(axis="y", labelcolor="black")

    # Combine legends
    lines = line1 + line2 + line3 + line4
    labels = [l.get_label() for l in lines]
    ax1.legend(lines, labels, loc="center right")

    # Add grid
    ax1.grid(True, alpha=0.3)

    # Set title
    plt.title("Batch-wise Training and Validation Metrics")

    # Adjust layout
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close()
    else:
        plt.show()


def plot_epoch_metrics(
    epoch_train_losses,
    epoch_val_losses,
    epoch_train_accuracies,
    epoch_val_accuracies,
    save_path=None,
):
    """
    Plot epoch-wise training and validation losses and accuracies using dual axes.

    Args:
        epoch_train_losses (list): List of training losses per epoch
        epoch_val_losses (list): List of validation losses per epoch
        epoch_train_accuracies (list): List of training accuracies per epoch
        epoch_val_accuracies (list): List of validation accuracies per epoch
        save_path (str, optional): Path to save the plot. If None, plot is displayed.
    """
    fig, ax1 = plt.subplots(figsize=(12, 6))

    # Create second y-axis
    ax2 = ax1.twinx()

    # Plot losses on primary y-axis
    line1 = ax1.plot(
        epoch_train_losses, label="Training Loss", color="blue", marker="o", alpha=0.7
    )
    line2 = ax1.plot(
        epoch_val_losses, label="Validation Loss", color="red", marker="o", alpha=0.7
    )
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss", color="black")
    ax1.tick_params(axis="y", labelcolor="black")

    # Plot accuracies on secondary y-axis
    line3 = ax2.plot(
        epoch_train_accuracies,
        label="Training Accuracy",
        color="green",
        marker="o",
        alpha=0.7,
    )
    line4 = ax2.plot(
        epoch_val_accuracies,
        label="Validation Accuracy",
        color="purple",
        marker="o",
        alpha=0.7,
    )
    ax2.set_ylabel("Accuracy", color="black")
    ax2.tick_params(axis="y", labelcolor="black")

    # Combine legends
    lines = line1 + line2 + line3 + line4
    labels = [l.get_label() for l in lines]
    ax1.legend(lines, labels, loc="center right")

    # Add grid
    ax1.grid(True, alpha=0.3)

    # Set title
    plt.title("Epoch-wise Training and Validation Metrics")

    # Adjust layout
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close()
    else:
        plt.show()
