import torch

def save_model(model, path="model.pth"):
    torch.save(model.state_dict(), path)
