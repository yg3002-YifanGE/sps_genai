# helper_lib/generator.py
"""
Image generator utilities for VAE.

Usage:
    from helper_lib.generator import generate_samples

    # 假设你已经训练好了 vae，并且 vae.latent_dim 存在：
    generate_samples(vae, device="cuda", num_samples=16, save_path="samples.png")
"""

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
    Randomly sample points in latent space and decode images with a trained VAE.

    Args:
        model: 训练完成的 VAE 模型，需包含 `latent_dim` 属性与 `decode(z)` 方法。
        device: 运行设备，比如 "cpu" 或 "cuda"。
        num_samples: 生成的图片数量（建议设为平方数以铺满网格，例如 9/16/25）。
        figsize: matplotlib 画布大小。
        save_path: 若提供，将把生成结果保存到该路径（例如 "samples.png"）。
        seed: 随机种子，便于复现实验结果。

    Returns:
        None
    """
    model = model.to(device)
    model.eval()

    if seed is not None:
        torch.manual_seed(seed)

    # 采样潜变量 z ~ N(0, I)
    if not hasattr(model, "latent_dim"):
        raise AttributeError("Model must have attribute `latent_dim` for sampling.")
    z = torch.randn(num_samples, model.latent_dim, device=device)

    # 使用解码器生成图片
    samples = model.decode(z).detach().cpu()  # (N, C, H, W) or (N, 1, H, W)

    # 处理通道：单通道用灰度显示
    if samples.dim() != 4:
        raise ValueError(f"Expected decoded samples with shape (N, C, H, W), got {samples.shape}")
    _, c, h, w = samples.shape

    # 建立 n x n 的展示网格
    n = int(math.ceil(math.sqrt(num_samples)))
    fig, axes = plt.subplots(n, n, figsize=figsize)
    if n == 1:
        axes = [axes]  # 退化为列表，统一处理
    else:
        axes = axes.flatten()

    # 将像素裁剪到 [0,1]（以防模型解码输出略越界）
    samples = samples.clamp(0.0, 1.0)

    # 逐格绘制
    for i in range(n * n):
        ax = axes[i]
        ax.axis("off")
        if i >= num_samples:
            continue  # 多余的格子留空
        img = samples[i]
        if c == 1:
            ax.imshow(img[0], cmap="gray", interpolation="nearest")
        elif c == 3:
            # (3, H, W) -> (H, W, 3)
            ax.imshow(img.permute(1, 2, 0), interpolation="nearest")
        else:
            # 其他通道数：只显示第一个通道
            ax.imshow(img[0], cmap="gray", interpolation="nearest")
            ax.set_title(f"ch={c} (show ch0)", fontsize=8)

    plt.tight_layout()

    if save_path is not None:
        plt.savefig(save_path, bbox_inches="tight")
        print(f"Saved samples to {save_path}")

    plt.show()
