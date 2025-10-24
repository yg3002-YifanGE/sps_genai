import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

# ---------- MNIST (1×28×28) ----------
def get_data_loader(data_dir: str, batch_size: int = 32, train: bool = True,
                    num_workers: int = 2, pin_memory: bool = True):
    """
    MNIST loader 
    Output images normalized to mean=0.5,std=0.5 so pixel range becomes ~[-1,1].
    """
    tfm = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,)),   # (mean,), (std,)
    ])
    ds = datasets.MNIST(root=data_dir, train=train, transform=tfm, download=True)
    return DataLoader(ds, batch_size=batch_size, shuffle=train,
                      num_workers=num_workers, pin_memory=pin_memory)


def get_mnist_loader(data_dir: str, batch_size: int = 32, train: bool = True,
                     num_workers: int = 2, pin_memory: bool = True):
    """
    Just a clearer alias for MNIST loader. Same behavior as get_data_loader.
    """
    return get_data_loader(
        data_dir=data_dir,
        batch_size=batch_size,
        train=train,
        num_workers=num_workers,
        pin_memory=pin_memory
    )


# ---------- CIFAR-10 (3×32×32 -> optional 64×64) ----------
def get_cifar10_loader(root: str = "data", batch_size: int = 128, train: bool = True,
                       resize_to_64: bool = True, augment: bool = True,
                       num_workers: int = 2, pin_memory: bool = True,
                       drop_last: bool = False):
    """
    Output CIFAR-10 DataLoader.
    - When resize_to_64=True, the 32×32 images are resized to 64×64 to align with your 64×64×3 CNN_A2; if your model takes 32×32 input, set it to False.
    - When augment=True, common lightweight augmentations (random horizontal flip + random crop) are applied to the training set. 
    """
    cifar10_mean = (0.4914, 0.4822, 0.4465)
    cifar10_std  = (0.2470, 0.2435, 0.2616)

    size = 64 if resize_to_64 else 32

    ops = []
    if resize_to_64:
        ops.append(transforms.Resize((size, size)))

    if train and augment:
        ops += [
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(size, padding=4)
        ]

    ops += [
        transforms.ToTensor(),
        transforms.Normalize(cifar10_mean, cifar10_std),
    ]
    tfm = transforms.Compose(ops)

    ds = datasets.CIFAR10(root=root, train=train, transform=tfm, download=True)
    return DataLoader(ds, batch_size=batch_size, shuffle=train,
                      num_workers=num_workers, pin_memory=False,
                      drop_last=drop_last)


def get_cifar10_loaders(root: str = "data", batch_size: int = 128,
                        resize_to_64: bool = True, augment: bool = True,
                        num_workers: int = 2, pin_memory: bool = True):
    train_loader = get_cifar10_loader(root=root, batch_size=batch_size, train=True,
                                      resize_to_64=resize_to_64, augment=augment,
                                      num_workers=num_workers, pin_memory=pin_memory)
    test_loader  = get_cifar10_loader(root=root, batch_size=batch_size, train=False,
                                      resize_to_64=resize_to_64, augment=False,
                                      num_workers=num_workers, pin_memory=pin_memory)
    return train_loader, test_loader
