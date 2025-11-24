"""
Offline training script for Energy-Based Model (EBM) on CIFAR-10.
This script trains an EBM on CIFAR-10 dataset and saves the weights.

Usage:
    python train_ebm_cifar10.py
"""

import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from pathlib import Path

from helper_lib.model import get_model
from helper_lib.trainer import train_ebm

def main():
    # Device configuration
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # Data preparation - CIFAR-10 normalized to [-1, 1]
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # Map to [-1, 1] for RGB
    ])

    train_dataset = torchvision.datasets.CIFAR10(
        root='./data',
        train=True,
        download=True,
        transform=transform
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=128,
        shuffle=True,
        num_workers=2
    )

    # Create EBM model for RGB images
    print("Creating EBM model for CIFAR-10 (RGB)...")
    model = get_model("EBM", num_channels=3)

    # Training hyperparameters
    epochs = 5  # Reduced for faster training
    lr = 1e-4
    alpha = 0.1  # Regularization weight
    steps = 30  # Langevin dynamics steps (reduced from 60 for speed)
    step_size = 10  # Step size for gradient descent on input
    noise = 0.005  # Noise std for Langevin dynamics

    print(f"\nTraining EBM on CIFAR-10 for {epochs} epochs...")
    print(f"Hyperparameters:")
    print(f"  Dataset: CIFAR-10 (32x32 RGB)")
    print(f"  Learning rate: {lr}")
    print(f"  Alpha (reg): {alpha}")
    print(f"  Langevin steps: {steps}")
    print(f"  Step size: {step_size}")
    print(f"  Noise std: {noise}")
    print()

    # Save directory
    save_dir = Path("data")
    save_dir.mkdir(exist_ok=True)
    
    # Train the model with epoch-by-epoch saving
    print("Note: Model will be saved after each epoch")
    
    trained_model, buffer = train_ebm(
        model=model,
        data_loader=train_loader,
        device=device,
        epochs=epochs,
        lr=lr,
        alpha=alpha,
        steps=steps,
        step_size=step_size,
        noise=noise,
        num_channels=3  # RGB
    )

    # Save the final trained model
    save_path = save_dir / "ebm_cifar10.pth"
    torch.save(trained_model.state_dict(), save_path)
    print(f"\n✅ EBM model (CIFAR-10) saved to: {save_path}")

    # Generate and display some samples
    print("\nGenerating sample images...")
    with torch.no_grad():
        # Get some samples from buffer
        sample_imgs = torch.cat(buffer.examples[:16], dim=0)
        # Rescale from [-1,1] to [0,1]
        sample_imgs = (sample_imgs + 1.0) / 2.0
        sample_imgs = sample_imgs.clamp(0.0, 1.0)

        # Save grid
        from torchvision.utils import save_image
        grid_path = save_dir / "ebm_cifar10_samples.png"
        save_image(sample_imgs, grid_path, nrow=4)
        print(f"Sample images saved to: {grid_path}")

    print("\n✅ Training complete!")
    print(f"To use in API, ensure {save_path} exists when running the server.")


if __name__ == "__main__":
    main()

