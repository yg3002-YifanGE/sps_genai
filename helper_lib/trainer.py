import torch
import torch.nn as nn
import torch.nn.functional as F

def train_model(model, data_loader, criterion, optimizer, device='cpu', epochs=10):
    model.to(device)
    model.train()
    for epoch in range(epochs):
        running_loss = 0.0
        for images, labels in data_loader:
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
        print(f"Epoch {epoch+1}/{epochs}, Loss: {running_loss/len(data_loader)}")
    return model

# VAE loss (重构 BCE + KL 散度)
def vae_loss(recon_x, x, mu, logvar, beta: float = 1.0):
    # 重构误差（像素级别的 BCE）
    bce = F.binary_cross_entropy(recon_x, x, reduction='sum')
    # KL 散度项
    kld = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return bce + beta * kld, bce, kld

def train_vae_model(model, data_loader, optimizer, device='cpu', epochs=10, beta: float = 1.0):
    model.to(device)
    model.train()
    for epoch in range(epochs):
        running_loss, running_bce, running_kld = 0.0, 0.0, 0.0
        count = 0
        for images, _ in data_loader:  # VAE 不用 labels
            images = images.to(device)

            optimizer.zero_grad()
            recon, mu, logvar = model(images)
            loss, bce, kld = vae_loss(recon, images, mu, logvar, beta)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            running_bce += bce.item()
            running_kld += kld.item()
            count += images.size(0)

        print(f"Epoch {epoch+1}/{epochs}, "
              f"Loss: {running_loss/count:.4f}, "
              f"BCE: {running_bce/count:.4f}, "
              f"KLD: {running_kld/count:.4f}")
    return model


# ====== GAN training ======
def train_gan(models,
              data_loader,
              device='cpu',
              epochs=10,
              lr=2e-4,
              beta1=0.5,
              latent_dim=100):
    """
    models: dict with {
        "generator": G,
        "discriminator": D
    }
    data_loader: MNIST loader (images already normalized to [-1,1] ideally)
    """

    G = models["generator"].to(device)
    D = models["discriminator"].to(device)

    # loss fn: Binary Cross Entropy
    criterion = nn.BCELoss()

    # separate optimizers
    optimizerD = torch.optim.Adam(D.parameters(), lr=lr, betas=(beta1, 0.999))
    optimizerG = torch.optim.Adam(G.parameters(), lr=lr, betas=(beta1, 0.999))

    for epoch in range(epochs):
        running_loss_D = 0.0
        running_loss_G = 0.0

        for real_imgs, _ in data_loader:
            real_imgs = real_imgs.to(device)  # shape (B,1,28,28)

            batch_size = real_imgs.size(0)

            # -----------------
            # 1. Train Discriminator
            # -----------------
            D.zero_grad(set_to_none=True)

            # real labels = 1, fake labels = 0
            real_labels = torch.ones(batch_size, 1, device=device)
            fake_labels = torch.zeros(batch_size, 1, device=device)

            # forward real batch
            out_real = D(real_imgs)
            loss_real = criterion(out_real, real_labels)

            # generate fake batch
            z = torch.randn(batch_size, latent_dim, device=device)
            fake_imgs = G(z)
            out_fake = D(fake_imgs.detach())
            loss_fake = criterion(out_fake, fake_labels)

            loss_D = loss_real + loss_fake
            loss_D.backward()
            optimizerD.step()

            # -----------------
            # 2. Train Generator
            # -----------------
            G.zero_grad(set_to_none=True)

            # goal: fool D -> labels should be 1
            out_fake_for_G = D(fake_imgs)
            loss_G = criterion(out_fake_for_G, real_labels)
            loss_G.backward()
            optimizerG.step()

            running_loss_D += loss_D.item()
            running_loss_G += loss_G.item()

        print(f"[Epoch {epoch+1}/{epochs}] "
              f"D_loss: {running_loss_D/len(data_loader):.4f} "
              f"G_loss: {running_loss_G/len(data_loader):.4f}")

    # return the updated dict
    models["generator"] = G
    models["discriminator"] = D
    return models


# ====== Energy-Based Model (EBM) training ======
def generate_ebm_samples(nn_energy_model, inp_imgs, steps, step_size, noise_std):
    """
    Generate low-energy samples using Langevin dynamics.
    This performs gradient descent on the input images to find low-energy states.
    """
    nn_energy_model.eval()

    for _ in range(steps):
        # Add noise
        with torch.no_grad():
            noise = torch.randn_like(inp_imgs) * noise_std
            inp_imgs = (inp_imgs + noise).clamp(-1.0, 1.0)

        inp_imgs.requires_grad_(True)

        # Compute energy and gradients with respect to input
        energy = nn_energy_model(inp_imgs)

        # Calculate gradient manually to control what gets differentiated
        grads, = torch.autograd.grad(energy, inp_imgs, grad_outputs=torch.ones_like(energy))

        # Apply gradient descent on input (moving to low energy states)
        with torch.no_grad():
            grads = grads.clamp(-0.03, 0.03)
            inp_imgs = (inp_imgs - step_size * grads).clamp(-1.0, 1.0)

    return inp_imgs.detach()


class EBMBuffer:
    """
    Buffer to store generated samples for efficient EBM training.
    Reuses previous samples to avoid costly Langevin dynamics from scratch.
    """
    def __init__(self, model, device, num_channels=3, buffer_size=8192):
        self.model = model
        self.device = device
        self.num_channels = num_channels
        # Initialize with random images
        self.examples = [torch.rand((1, num_channels, 32, 32), device=self.device) * 2 - 1 for _ in range(128)]

    def sample_new_exmps(self, steps, step_size, noise):
        import numpy as np
        import random
        n_new = np.random.binomial(128, 0.05)

        # Generate ~5% new random images
        new_rand_imgs = torch.rand((n_new, self.num_channels, 32, 32), device=self.device) * 2 - 1

        # Sample the rest from buffer
        old_imgs = torch.cat(random.choices(self.examples, k=128 - n_new), dim=0)

        inp_imgs = torch.cat([new_rand_imgs, old_imgs], dim=0)

        # Run Langevin dynamics
        new_imgs = generate_ebm_samples(self.model, inp_imgs, steps, step_size, noise)

        # Update buffer
        self.examples = list(torch.split(new_imgs, 1, dim=0)) + self.examples
        self.examples = self.examples[:8192]

        return new_imgs


def train_ebm(model, data_loader, device='cpu', epochs=10, lr=1e-4, 
              alpha=0.1, steps=60, step_size=10, noise=0.005, num_channels=3):
    """
    Train an Energy-Based Model using contrastive divergence.
    
    Args:
        model: EnergyModel instance
        data_loader: DataLoader for CIFAR-10 (images should be normalized to [-1,1])
        device: computation device
        epochs: number of training epochs
        lr: learning rate
        alpha: regularization weight
        steps: number of Langevin dynamics steps
        step_size: step size for gradient descent on input
        noise: noise std for Langevin dynamics
        num_channels: number of image channels (3 for RGB CIFAR-10)
    """
    model.to(device)
    model.train()
    
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, betas=(0.0, 0.999))
    buffer = EBMBuffer(model, device=device, num_channels=num_channels)

    for epoch in range(epochs):
        running_loss = 0.0
        running_cdiv = 0.0
        running_real = 0.0
        running_fake = 0.0
        count = 0

        for images, _ in data_loader:
            # Add noise to real images
            real_imgs = images.to(device)
            real_imgs = real_imgs + torch.randn_like(real_imgs) * noise
            real_imgs = torch.clamp(real_imgs, -1.0, 1.0)

            # Sample fake images from buffer
            fake_imgs = buffer.sample_new_exmps(steps=steps, step_size=step_size, noise=noise)

            # Concatenate and compute energies
            inp_imgs = torch.cat([real_imgs, fake_imgs], dim=0)
            inp_imgs = inp_imgs.clone().detach().requires_grad_(False)

            out_scores = model(inp_imgs)
            real_out, fake_out = torch.split(out_scores, [real_imgs.size(0), fake_imgs.size(0)], dim=0)

            # Contrastive divergence loss: increase energy on fake, decrease on real
            cdiv_loss = real_out.mean() - fake_out.mean()
            reg_loss = alpha * (real_out.pow(2).mean() + fake_out.pow(2).mean())
            loss = cdiv_loss + reg_loss

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.1)
            optimizer.step()

            running_loss += loss.item()
            running_cdiv += cdiv_loss.item()
            running_real += real_out.mean().item()
            running_fake += fake_out.mean().item()
            count += 1

        print(f"[Epoch {epoch+1}/{epochs}] "
              f"Loss: {running_loss/count:.4f}, "
              f"CDivLoss: {running_cdiv/count:.4f}, "
              f"Real: {running_real/count:.4f}, "
              f"Fake: {running_fake/count:.4f}")

    return model, buffer


# ====== Diffusion Model training ======
def linear_diffusion_schedule(diffusion_times, min_rate=1e-4, max_rate=0.02):
    """Linear diffusion schedule for noise addition"""
    diffusion_times = diffusion_times.to(dtype=torch.float32)
    betas = min_rate + diffusion_times * (max_rate - min_rate)
    alphas = 1.0 - betas
    alpha_bars = torch.cumprod(alphas, dim=0)
    signal_rates = torch.sqrt(alpha_bars)
    noise_rates = torch.sqrt(1.0 - alpha_bars)
    return noise_rates, signal_rates


def cosine_diffusion_schedule(diffusion_times):
    """Cosine diffusion schedule - smoother noise addition"""
    import math
    signal_rates = torch.cos(diffusion_times * math.pi / 2)
    noise_rates = torch.sin(diffusion_times * math.pi / 2)
    return noise_rates, signal_rates


def offset_cosine_diffusion_schedule(diffusion_times, min_signal_rate=0.02, max_signal_rate=0.95):
    """Offset cosine schedule - avoids extreme values"""
    import math
    original_shape = diffusion_times.shape
    diffusion_times_flat = diffusion_times.flatten()

    start_angle = torch.acos(torch.tensor(max_signal_rate, dtype=torch.float32, device=diffusion_times.device))
    end_angle = torch.acos(torch.tensor(min_signal_rate, dtype=torch.float32, device=diffusion_times.device))

    diffusion_angles = start_angle + diffusion_times_flat * (end_angle - start_angle)

    signal_rates = torch.cos(diffusion_angles).reshape(original_shape)
    noise_rates = torch.sin(diffusion_angles).reshape(original_shape)

    return noise_rates, signal_rates


class DiffusionModelWrapper(nn.Module):
    """
    Wrapper for diffusion model that handles training and generation.
    """
    def __init__(self, unet_model, schedule_fn):
        super().__init__()
        self.network = unet_model
        self.schedule_fn = schedule_fn
        self.normalizer_mean = 0.0
        self.normalizer_std = 1.0

    def set_normalizer(self, mean, std):
        self.normalizer_mean = mean
        self.normalizer_std = std

    def denormalize(self, x):
        return torch.clamp(x * self.normalizer_std + self.normalizer_mean, 0.0, 1.0)

    def denoise(self, noisy_images, noise_rates, signal_rates):
        self.network.eval()
        with torch.no_grad():
            pred_noises = self.network(noisy_images, noise_rates ** 2)
            pred_images = (noisy_images - noise_rates * pred_noises) / signal_rates
        return pred_noises, pred_images

    def reverse_diffusion(self, initial_noise, diffusion_steps):
        """Generate images by reversing the diffusion process"""
        step_size = 1.0 / diffusion_steps
        current_images = initial_noise

        for step in range(diffusion_steps):
            t = torch.ones((initial_noise.shape[0], 1, 1, 1), device=initial_noise.device) * (1 - step * step_size)
            noise_rates, signal_rates = self.schedule_fn(t)
            pred_noises, pred_images = self.denoise(current_images, noise_rates, signal_rates)

            next_diffusion_times = t - step_size
            next_noise_rates, next_signal_rates = self.schedule_fn(next_diffusion_times)
            current_images = next_signal_rates * pred_images + next_noise_rates * pred_noises

        return pred_images

    def generate(self, num_images, diffusion_steps, image_size=32):
        """Generate new images from random noise"""
        device = next(self.parameters()).device
        initial_noise = torch.randn((num_images, self.network.num_channels, image_size, image_size), device=device)
        with torch.no_grad():
            return self.denormalize(self.reverse_diffusion(initial_noise, diffusion_steps))


def train_diffusion(model_wrapper, data_loader, device='cpu', epochs=10, lr=1e-3):
    """
    Train a diffusion model.
    
    Args:
        model_wrapper: DiffusionModelWrapper instance
        data_loader: DataLoader for training data
        device: computation device
        epochs: number of training epochs
        lr: learning rate
    """
    model_wrapper.to(device)
    model_wrapper.network.train()

    optimizer = torch.optim.AdamW(model_wrapper.network.parameters(), lr=lr, weight_decay=1e-4)
    criterion = nn.L1Loss()

    for epoch in range(epochs):
        running_loss = 0.0
        count = 0

        for images, _ in data_loader:
            images = images.to(device)
            
            # Normalize images
            images = (images - model_wrapper.normalizer_mean) / model_wrapper.normalizer_std
            noises = torch.randn_like(images)

            # Sample random diffusion times for each image in the batch
            diffusion_times = torch.rand((images.size(0), 1, 1, 1), device=images.device)

            # Get noise and signal rates from the diffusion schedule
            noise_rates, signal_rates = model_wrapper.schedule_fn(diffusion_times)
            noisy_images = signal_rates * images + noise_rates * noises

            # Predict the noise using the network
            pred_noises = model_wrapper.network(noisy_images, noise_rates ** 2)

            # Compute loss between predicted and true noise
            loss = criterion(pred_noises, noises)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            count += 1

        print(f"[Epoch {epoch+1}/{epochs}] Loss: {running_loss/count:.4f}")

    return model_wrapper
