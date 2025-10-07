# train_cifar10
import torch
import torch.nn as nn
import torch.optim as optim

from helper_lib.data_loader import get_cifar10_loaders
from helper_lib.model import get_model
from helper_lib.trainer import train_model

def evaluate(model, data_loader, device="cpu"):
    model.eval()
    correct = total = 0
    with torch.no_grad():
        for images, labels in data_loader:
            images, labels = images.to(device), labels.to(device)
            logits = model(images)
            pred = logits.argmax(dim=1)
            correct += (pred == labels).sum().item()
            total += labels.size(0)
    return correct / total

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    train_loader, test_loader = get_cifar10_loaders(
        root="data", batch_size=128, resize_to_64=True, augment=True
    )

    model = get_model("CNN_A2").to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    # 5 epoch
    model = train_model(model, train_loader, criterion, optimizer, device=device, epochs=5)

    # evaluation
    acc = evaluate(model, test_loader, device=device)
    print(f"Test Accuracy: {acc:.4%}")

    # save weight
    save_path = "assignment2/cnn_a2_cifar10.pt"
    torch.save(model.state_dict(), save_path)
    print(f"Saved weights to {save_path}")

if __name__ == "__main__":
    main()
