import math
from typing import Optional, Tuple

import torch
import matplotlib.pyplot as plt

@torch.no_grad()
def generate_samples(
    model: torch.nn.Module,
    device: str = "cpu",
    num_samples: int = 16,
    figsize: Tuple[int, int] = (6, 6),
    save_path: Optional[str] = None,
    seed: Optional[int] = None,
):
    """
    VAE sampler (保持你原来的逻辑不变)
    """
    model = model.to(device)
    model.eval()

    if seed is not None:
        torch.manual_seed(seed)

    if not hasattr(model, "latent_dim"):
        raise AttributeError("Model must have attribute `latent_dim` for sampling.")
    z = torch.randn(num_samples, model.latent_dim, device=device)

    samples = model.decode(z).detach().cpu()

    if samples.dim() != 4:
        raise ValueError(f"Expected decoded samples with shape (N, C, H, W), got {samples.shape}")
    _, c, h, w = samples.shape

    n = int(math.ceil(math.sqrt(num_samples)))
    fig, axes = plt.subplots(n, n, figsize=figsize)
    if n == 1:
        axes = [axes]
    else:
        axes = axes.flatten()

    samples = samples.clamp(0.0, 1.0)

    for i in range(n * n):
        ax = axes[i]
        ax.axis("off")
        if i >= num_samples:
            continue
        img = samples[i]
        if c == 1:
            ax.imshow(img[0], cmap="gray", interpolation="nearest")
        elif c == 3:
            ax.imshow(img.permute(1, 2, 0), interpolation="nearest")
        else:
            ax.imshow(img[0], cmap="gray", interpolation="nearest")
            ax.set_title(f"ch={c} (show ch0)", fontsize=8)

    plt.tight_layout()
    if save_path is not None:
        plt.savefig(save_path, bbox_inches="tight")
        print(f"Saved samples to {save_path}")
    plt.show()


@torch.no_grad()
def generate_gan_samples(
    generator_model: torch.nn.Module,
    device: str = "cpu",
    num_samples: int = 16,
    latent_dim: int = 100,
    figsize: Tuple[int, int] = (6, 6),
    save_path: Optional[str] = None,
    seed: Optional[int] = None,
):
    """
    GAN sampler:
    - generator_model: trained GANGenerator (outputs tanh -> [-1,1])
    - returns nothing, but plots a grid and (optionally) saves it
    """
    generator_model = generator_model.to(device)
    generator_model.eval()

    if seed is not None:
        torch.manual_seed(seed)

    # sample latent noise
    z = torch.randn(num_samples, latent_dim, device=device)
    fake_imgs = generator_model(z).detach().cpu()  # (N,1,28,28), range [-1,1] because tanh

    # rescale [-1,1] -> [0,1]
    fake_imgs = (fake_imgs + 1.0) / 2.0
    fake_imgs = fake_imgs.clamp(0.0, 1.0)

    n = int(math.ceil(math.sqrt(num_samples)))
    fig, axes = plt.subplots(n, n, figsize=figsize)
    if n == 1:
        axes = [axes]
    else:
        axes = axes.flatten()

    for i in range(n * n):
        ax = axes[i]
        ax.axis("off")
        if i >= num_samples:
            continue
        img = fake_imgs[i][0]  # single channel
        ax.imshow(img, cmap="gray", interpolation="nearest")

    plt.tight_layout()
    if save_path is not None:
        plt.savefig(save_path, bbox_inches="tight")
        print(f"Saved GAN samples to {save_path}")
    plt.show()


@torch.no_grad()
def generate_ebm_samples(
    energy_model: torch.nn.Module,
    device: str = "cpu",
    num_samples: int = 16,
    steps: int = 256,
    step_size: float = 10.0,
    noise_std: float = 0.01,
    figsize: Tuple[int, int] = (6, 6),
    save_path: Optional[str] = None,
    seed: Optional[int] = None,
):
    """
    Generate samples from an Energy-Based Model using Langevin dynamics.
    
    Args:
        energy_model: Trained EnergyModel
        device: Device to run on
        num_samples: Number of samples to generate
        steps: Number of Langevin dynamics steps
        step_size: Step size for gradient descent
        noise_std: Standard deviation of noise to add at each step
        figsize: Figure size for plotting
        save_path: Optional path to save the plot
        seed: Random seed for reproducibility
    """
    energy_model = energy_model.to(device)
    energy_model.eval()

    if seed is not None:
        torch.manual_seed(seed)

    # Start from random noise [-1, 1]
    x = torch.rand((num_samples, 1, 32, 32), device=device) * 2 - 1

    # Run Langevin dynamics to find low-energy states
    for _ in range(steps):
        # Add noise
        noise = torch.randn_like(x) * noise_std
        x = (x + noise).clamp(-1.0, 1.0)

        x.requires_grad_(True)

        # Compute energy and gradients
        energy = energy_model(x)
        grads, = torch.autograd.grad(energy, x, grad_outputs=torch.ones_like(energy))

        # Gradient descent on input (move to low energy)
        with torch.no_grad():
            grads = grads.clamp(-0.03, 0.03)
            x = (x - step_size * grads).clamp(-1.0, 1.0)

    samples = x.detach().cpu()
    
    # Rescale from [-1,1] to [0,1]
    samples = (samples + 1.0) / 2.0
    samples = samples.clamp(0.0, 1.0)

    n = int(math.ceil(math.sqrt(num_samples)))
    fig, axes = plt.subplots(n, n, figsize=figsize)
    if n == 1:
        axes = [axes]
    else:
        axes = axes.flatten()

    for i in range(n * n):
        ax = axes[i]
        ax.axis("off")
        if i >= num_samples:
            continue
        img = samples[i][0]  # single channel
        ax.imshow(img, cmap="gray", interpolation="nearest")

    plt.tight_layout()
    if save_path is not None:
        plt.savefig(save_path, bbox_inches="tight")
        print(f"Saved EBM samples to {save_path}")
    plt.show()


@torch.no_grad()
def generate_diffusion_samples(
    diffusion_model_wrapper,
    device: str = "cpu",
    num_samples: int = 16,
    diffusion_steps: int = 50,
    image_size: int = 32,
    figsize: Tuple[int, int] = (6, 6),
    save_path: Optional[str] = None,
    seed: Optional[int] = None,
):
    """
    Generate samples from a Diffusion Model.
    
    Args:
        diffusion_model_wrapper: Trained DiffusionModelWrapper
        device: Device to run on
        num_samples: Number of samples to generate
        diffusion_steps: Number of reverse diffusion steps
        image_size: Size of generated images
        figsize: Figure size for plotting
        save_path: Optional path to save the plot
        seed: Random seed for reproducibility
    """
    diffusion_model_wrapper = diffusion_model_wrapper.to(device)
    diffusion_model_wrapper.eval()

    if seed is not None:
        torch.manual_seed(seed)

    # Generate samples
    samples = diffusion_model_wrapper.generate(
        num_images=num_samples, 
        diffusion_steps=diffusion_steps,
        image_size=image_size
    ).cpu()

    n = int(math.ceil(math.sqrt(num_samples)))
    fig, axes = plt.subplots(n, n, figsize=figsize)
    if n == 1:
        axes = [axes]
    else:
        axes = axes.flatten()

    for i in range(n * n):
        ax = axes[i]
        ax.axis("off")
        if i >= num_samples:
            continue
        img = samples[i]
        
        # Handle different channel counts
        if img.shape[0] == 1:  # grayscale
            ax.imshow(img[0], cmap="gray", interpolation="nearest")
        elif img.shape[0] == 3:  # RGB
            ax.imshow(img.permute(1, 2, 0), interpolation="nearest")
        else:
            ax.imshow(img[0], cmap="gray", interpolation="nearest")

    plt.tight_layout()
    if save_path is not None:
        plt.savefig(save_path, bbox_inches="tight")
        print(f"Saved Diffusion samples to {save_path}")
    plt.show()
