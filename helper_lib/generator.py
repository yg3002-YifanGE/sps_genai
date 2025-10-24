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
