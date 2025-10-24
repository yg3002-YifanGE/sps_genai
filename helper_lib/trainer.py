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
