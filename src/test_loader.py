from torchvision import datasets, transforms
from torch.utils.data import DataLoader


def get_cifar10_test_loader(batch_size: int = 256):
    transform = transforms.Compose([
        transforms.Resize((96, 96)),
        transforms.ToTensor(),
    ])

    dataset = datasets.CIFAR10(
        root="./data",
        train=False,
        download=True,
        transform=transform
    )

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False
    )

    return dataset, loader