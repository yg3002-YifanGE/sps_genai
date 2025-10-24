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


# ====== get_model ======
def get_model(model_name: str, **kwargs):
    """
    model_name should be one of:
    - "CNN_A2"
    - "VAE"
    - "GAN"
    kwargs can include:
    - num_classes for CNN_A2
    - latent_dim for VAE or GANGenerator
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
    else:
        raise ValueError(f"Unknown model name: {model_name}")
