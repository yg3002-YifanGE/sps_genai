import torch
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
