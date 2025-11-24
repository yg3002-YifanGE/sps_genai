import torch
import torch.nn as nn
import torch.nn.functional as F

class CNN_A2(nn.Module):
    """
    input: (N, 3, 64, 64)
    structure:
      Conv(3->16, k3,s1,p1) + ReLU + MaxPool(2,2)
      Conv(16->32, k3,s1,p1) + ReLU + MaxPool(2,2)
      Flatten
      FC(32*16*16 -> 100) + ReLU
      FC(100 -> 10)
    """
    def __init__(self, num_classes: int = 10):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)
        self.pool  = nn.MaxPool2d(kernel_size=2, stride=2)
        self.fc1 = nn.Linear(32 * 16 * 16, 100)
        self.fc2 = nn.Linear(100, num_classes)

    def forward(self, x):
        x = F.relu(self.conv1(x))   # (N,16,64,64)
        x = self.pool(x)            # (N,16,32,32)
        x = F.relu(self.conv2(x))   # (N,32,32,32)
        x = self.pool(x)            # (N,32,16,16)
        x = x.view(x.size(0), -1)   # (N,32*16*16)
        x = F.relu(self.fc1(x))     # (N,100)
        x = self.fc2(x)             # (N,10)
        return x


# ====== VAE ======
class VAE(nn.Module):
    def __init__(self, latent_dim: int = 20):
        super().__init__()
        self.latent_dim = latent_dim
        # Encoder
        self.enc = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=4, stride=2, padding=1),  # (N,32,14,14)
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1), # (N,64,7,7)
            nn.ReLU(),
        )
        self.enc_flat_dim = 64*7*7
        self.fc_mu = nn.Linear(self.enc_flat_dim, latent_dim)
        self.fc_logvar = nn.Linear(self.enc_flat_dim, latent_dim)

        # Decoder
        self.fc_dec = nn.Linear(latent_dim, self.enc_flat_dim)
        self.dec = nn.Sequential(
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 1, kernel_size=4, stride=2, padding=1),
            nn.Sigmoid()
        )

    def encode(self, x):
        h = self.enc(x).view(x.size(0), -1)
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        return mu, logvar

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        h = self.fc_dec(z).view(-1, 64, 7, 7)
        return self.dec(h)

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        recon = self.decode(z)
        return recon, mu, logvar


# ====== GAN ======
class GANGenerator(nn.Module):
    """
    Generator:
    - input: (N, 100)
    - output: (N, 1, 28, 28) in [-1, 1] via Tanh
    """
    def __init__(self, latent_dim: int = 100):
        super().__init__()
        self.latent_dim = latent_dim

        # fully connected to 7x7x128
        self.fc = nn.Linear(latent_dim, 128 * 7 * 7)

        # upsample blocks
        self.deconv1 = nn.ConvTranspose2d(
            in_channels=128,
            out_channels=64,
            kernel_size=4,
            stride=2,
            padding=1
        )
        self.bn1 = nn.BatchNorm2d(64)

        self.deconv2 = nn.ConvTranspose2d(
            in_channels=64,
            out_channels=1,
            kernel_size=4,
            stride=2,
            padding=1
        )
        # no BN on final layer, then Tanh in forward()

    def forward(self, z):
        # z: (N, latent_dim)
        x = self.fc(z)                       # (N, 128*7*7)
        x = x.view(-1, 128, 7, 7)            # (N,128,7,7)
        x = F.relu(self.bn1(self.deconv1(x)))# (N,64,14,14)
        x = torch.tanh(self.deconv2(x))      # (N,1,28,28)
        return x


class GANDiscriminator(nn.Module):
    """
    Discriminator:
    - input: (N, 1, 28, 28)
    - output: (N, 1) probability real (0-1)
    """
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(
            in_channels=1,
            out_channels=64,
            kernel_size=4,
            stride=2,
            padding=1
        )
        self.conv2 = nn.Conv2d(
            in_channels=64,
            out_channels=128,
            kernel_size=4,
            stride=2,
            padding=1
        )
        self.bn2 = nn.BatchNorm2d(128)

        self.fc = nn.Linear(128 * 7 * 7, 1)

    def forward(self, x):
        # x: (N,1,28,28)
        x = F.leaky_relu(self.conv1(x), 0.2)          # (N,64,14,14)
        x = F.leaky_relu(self.bn2(self.conv2(x)), 0.2)# (N,128,7,7)
        x = x.view(x.size(0), -1)                     # (N,128*7*7)
        x = torch.sigmoid(self.fc(x))                 # (N,1)
        return x


# ====== Energy-Based Model (EBM) ======
def swish(x):
    """Swish activation function: x * sigmoid(x)"""
    return x * torch.sigmoid(x)


class EnergyModel(nn.Module):
    """
    Energy-based model that maps images to scalar energy values.
    Used for generating images by sampling low-energy states.
    Input: (N, 3, 32, 32) - RGB CIFAR-10 images
    Output: (N, 1) - scalar energy for each image
    """
    def __init__(self, num_channels=3):
        super().__init__()
        self.num_channels = num_channels
        self.conv1 = nn.Conv2d(num_channels, 16, kernel_size=5, stride=2, padding=2)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1)
        self.conv4 = nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1)
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(64 * 2 * 2, 64)
        self.fc2 = nn.Linear(64, 1)

    def forward(self, x):
        x = swish(self.conv1(x))
        x = swish(self.conv2(x))
        x = swish(self.conv3(x))
        x = swish(self.conv4(x))
        x = self.flatten(x)
        x = swish(self.fc1(x))
        return self.fc2(x)


# ====== Diffusion Model Components ======
class SinusoidalEmbedding(nn.Module):
    """Time step embedding using sinusoidal functions"""
    def __init__(self, num_frequencies: int = 16):
        super().__init__()
        self.num_frequencies = num_frequencies
        import math
        frequencies = torch.exp(torch.linspace(math.log(1.0), math.log(1000.0), num_frequencies))
        self.register_buffer("angular_speeds", 2.0 * math.pi * frequencies.view(1, 1, 1, -1))

    def forward(self, x):
        """
        x: Tensor of shape (B, 1, 1, 1)
        returns: Tensor of shape (B, 1, 1, 2 * num_frequencies)
        """
        x = x.expand(-1, 1, 1, self.num_frequencies)
        sin_part = torch.sin(self.angular_speeds * x)
        cos_part = torch.cos(self.angular_speeds * x)
        return torch.cat([sin_part, cos_part], dim=-1)


class ResidualBlock(nn.Module):
    """Residual block for UNet"""
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.needs_projection = in_channels != out_channels
        if self.needs_projection:
            self.proj = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        else:
            self.proj = nn.Identity()
        self.norm = nn.BatchNorm2d(in_channels, affine=False)
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)

    def forward(self, x):
        residual = self.proj(x)
        x = swish(self.conv1(x))
        x = self.conv2(x)
        return x + residual


class DownBlock(nn.Module):
    """Downsampling block for UNet"""
    def __init__(self, width: int, block_depth: int, in_channels: int):
        super().__init__()
        self.blocks = nn.ModuleList()
        for i in range(block_depth):
            self.blocks.append(ResidualBlock(in_channels, width))
            in_channels = width
        self.pool = nn.AvgPool2d(kernel_size=2)

    def forward(self, x, skips):
        for block in self.blocks:
            x = block(x)
            skips.append(x)
        x = self.pool(x)
        return x


class UpBlock(nn.Module):
    """Upsampling block for UNet"""
    def __init__(self, width: int, block_depth: int, in_channels: int):
        super().__init__()
        self.blocks = nn.ModuleList()
        for _ in range(block_depth):
            self.blocks.append(ResidualBlock(in_channels + width, width))
            in_channels = width

    def forward(self, x, skips):
        x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=False)
        for block in self.blocks:
            skip = skips.pop()
            x = torch.cat([x, skip], dim=1)
            x = block(x)
        return x


class UNet(nn.Module):
    """
    UNet architecture for diffusion models.
    Takes noisy images and noise variance as input, predicts the noise.
    Default: RGB CIFAR-10 images (3, 32, 32)
    """
    def __init__(self, image_size: int = 32, num_channels: int = 3, embedding_dim: int = 32):
        super().__init__()
        self.initial = nn.Conv2d(num_channels, 32, kernel_size=1)
        self.num_channels = num_channels
        self.image_size = image_size
        self.embedding_dim = embedding_dim
        self.embedding = SinusoidalEmbedding(num_frequencies=16)
        self.embedding_proj = nn.Conv2d(embedding_dim, 32, kernel_size=1)

        self.down1 = DownBlock(32, in_channels=64, block_depth=2)
        self.down2 = DownBlock(64, in_channels=32, block_depth=2)
        self.down3 = DownBlock(96, in_channels=64, block_depth=2)

        self.mid1 = ResidualBlock(in_channels=96, out_channels=128)
        self.mid2 = ResidualBlock(in_channels=128, out_channels=128)

        self.up1 = UpBlock(96, in_channels=128, block_depth=2)
        self.up2 = UpBlock(64, block_depth=2, in_channels=96)
        self.up3 = UpBlock(32, block_depth=2, in_channels=64)

        self.final = nn.Conv2d(32, num_channels, kernel_size=1)
        nn.init.zeros_(self.final.weight)

    def forward(self, noisy_images, noise_variances):
        skips = []
        x = self.initial(noisy_images)
        noise_emb = self.embedding(noise_variances)  # shape: (B, 1, 1, 32)
        # Upsample to match image size
        noise_emb = F.interpolate(noise_emb.permute(0, 3, 1, 2), size=(self.image_size, self.image_size), mode='nearest')
        x = torch.cat([x, noise_emb], dim=1)

        x = self.down1(x, skips)
        x = self.down2(x, skips)
        x = self.down3(x, skips)

        x = self.mid1(x)
        x = self.mid2(x)

        x = self.up1(x, skips)
        x = self.up2(x, skips)
        x = self.up3(x, skips)

        return self.final(x)


# ====== get_model ======
def get_model(model_name: str, **kwargs):
    """
    model_name should be one of:
    - "CNN_A2"
    - "VAE"
    - "GAN"
    - "EBM" (Energy-Based Model)
    - "Diffusion" (returns UNet for diffusion)
    kwargs can include:
    - num_classes for CNN_A2
    - latent_dim for VAE or GANGenerator
    - image_size, num_channels for Diffusion UNet
    """
    name = model_name.upper()
    if name == "CNN_A2":
        return CNN_A2(**kwargs)
    elif name == "VAE":
        return VAE(**kwargs)
    elif name == "GAN":
        # for GAN we return both generator and discriminator
        latent_dim = kwargs.get("latent_dim", 100)
        G = GANGenerator(latent_dim=latent_dim)
        D = GANDiscriminator()
        return {"generator": G, "discriminator": D}
    elif name == "EBM":
        num_channels = kwargs.get("num_channels", 3)  # Default to RGB (CIFAR-10)
        return EnergyModel(num_channels=num_channels)
    elif name == "DIFFUSION":
        image_size = kwargs.get("image_size", 32)
        num_channels = kwargs.get("num_channels", 3)  # Default to RGB (CIFAR-10)
        embedding_dim = kwargs.get("embedding_dim", 32)
        return UNet(image_size=image_size, num_channels=num_channels, embedding_dim=embedding_dim)
    else:
        raise ValueError(f"Unknown model name: {model_name}")
