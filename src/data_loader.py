from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset


def get_the_cifar10_train_loader(batch_size: int = 256, subset_size: int | None = None):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])

    dataset = datasets.CIFAR10(
        root="./data",
        train=True,
        download=True,
        transform=transform
    )

    if subset_size is not None:
        dataset = Subset(dataset, list(range(subset_size)))

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False
    )

    return dataset, loader