import torch
from pathlib import Path

from helper_lib.model import get_model
from helper_lib.data_loader import get_data_loader  # or get_mnist_loader if you added alias
from helper_lib.trainer import train_gan

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # 1. get model
    models = get_model("GAN", latent_dim=100)
    # models = {"generator": G, "discriminator": D}

    # 2. get MNIST dataloader
    data_dir = "data"
    batch_size = 128
    train_loader = get_data_loader(data_dir=data_dir, batch_size=batch_size, train=True)

    # 3. train GAN
    trained = train_gan(
        models=models,
        data_loader=train_loader,
        device=device,
        epochs=10,
        lr=2e-4,
        beta1=0.5,
        latent_dim=100,
    )

    G_trained = trained["generator"].cpu().eval()

    # 4. get Generator's weights
    out_path = Path("data") / "gan_G.pth"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(G_trained.state_dict(), out_path)

    print(f"Saved trained generator weights to {out_path}")

if __name__ == "__main__":
    main()
