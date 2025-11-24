"""
Offline training script for Diffusion Model on CIFAR-10.
This script trains a Diffusion model on CIFAR-10 dataset and saves the weights.

Usage:
    python train_diffusion_cifar10.py
"""

import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from pathlib import Path

from helper_lib.model import get_model
from helper_lib.trainer import (
    DiffusionModelWrapper, 
    offset_cosine_diffusion_schedule,
    train_diffusion
)

def main():
    # Device configuration
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # Data preparation - CIFAR-10
    transform = transforms.Compose([
        transforms.ToTensor(),
        # No normalization here - will be done in training
    ])

    train_dataset = torchvision.datasets.CIFAR10(
        root='./data',
        train=True,
        download=True,
        transform=transform
    )

    test_dataset = torchvision.datasets.CIFAR10(
        root='./data',
        train=False,
        download=True,
        transform=transform
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=128,
        shuffle=True,
        num_workers=2
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=128,
        shuffle=False,
        num_workers=2
    )

    # Create Diffusion model (UNet) for RGB images
    print("Creating Diffusion model (UNet) for CIFAR-10 (RGB)...")
    unet = get_model("Diffusion", image_size=32, num_channels=3, embedding_dim=32)
    
    # Wrap in DiffusionModelWrapper
    diffusion_model = DiffusionModelWrapper(unet, offset_cosine_diffusion_schedule)

    # Calculate normalization statistics for RGB
    print("Calculating normalization statistics...")
    mean = torch.zeros(3)
    std = torch.zeros(3)
    total_samples = 0

    for imgs, _ in train_loader:
        batch_size = imgs.size(0)
        imgs_flat = imgs.view(batch_size, 3, -1)
        batch_mean = imgs_flat.mean(dim=(0, 2))
        batch_std = imgs_flat.std(dim=(0, 2))
        mean += batch_mean * batch_size
        std += batch_std * batch_size
        total_samples += batch_size

    mean /= total_samples
    std /= total_samples
    print(f"Normalization - Mean: {mean.tolist()}, Std: {std.tolist()}")

    # Set normalizer
    mean = mean.reshape(1, 3, 1, 1).to(device)
    std = std.reshape(1, 3, 1, 1).to(device)
    diffusion_model.set_normalizer(mean, std)

    # Training hyperparameters
    epochs = 5  # Reduced for faster training
    lr = 1e-3

    print(f"\nTraining Diffusion model on CIFAR-10 for {epochs} epochs...")
    print(f"Hyperparameters:")
    print(f"  Dataset: CIFAR-10 (32x32 RGB)")
    print(f"  Learning rate: {lr}")
    print(f"  Image size: 32x32")
    print(f"  Channels: 3 (RGB)")
    print()

    # Train the model
    trained_model = train_diffusion(
        model_wrapper=diffusion_model,
        data_loader=train_loader,
        device=device,
        epochs=epochs,
        lr=lr
    )

    # Save the trained model
    save_dir = Path("data")
    save_dir.mkdir(exist_ok=True)
    save_path = save_dir / "diffusion_cifar10.pth"

    checkpoint = {
        "model_state": trained_model.network.state_dict(),
        "normalizer_mean": trained_model.normalizer_mean,
        "normalizer_std": trained_model.normalizer_std,
    }

    torch.save(checkpoint, save_path)
    print(f"\n✅ Diffusion model (CIFAR-10) saved to: {save_path}")

    # Generate and display some samples
    print("\nGenerating sample images...")
    trained_model.eval()
    with torch.no_grad():
        samples = trained_model.generate(
            num_images=16,
            diffusion_steps=50,
            image_size=32
        ).cpu()

        # Save grid
        from torchvision.utils import save_image
        grid_path = save_dir / "diffusion_cifar10_samples.png"
        save_image(samples, grid_path, nrow=4)
        print(f"Sample images saved to: {grid_path}")

    print("\n✅ Training complete!")
    print(f"To use in API, ensure {save_path} exists when running the server.")


if __name__ == "__main__":
    main()

