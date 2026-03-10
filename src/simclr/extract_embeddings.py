import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from simclr.simclr_model import SimCLR


def get_device():
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def load_trained_simclr(model_path: str):
    device = get_device()
    model = SimCLR(projection_dim=128).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    return model, device


def get_cifar10_plain_loader(train=True, batch_size=256):
    transform = transforms.Compose([
        transforms.ToTensor(),
    ])

    dataset = datasets.CIFAR10(
        root="./data",
        train=train,
        download=True,
        transform=transform
    )

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0
    )

    return dataset, loader


def extract_embeddings(model_path: str, train=True):
    model, device = load_trained_simclr(model_path)
    dataset, loader = get_cifar10_plain_loader(train=train)

    all_embeddings = []

    with torch.no_grad():
        for images, _ in loader:
            images = images.to(device)
            h, _ = model(images)
            h = F.normalize(h, dim=1)
            all_embeddings.append(h.cpu())

    embeddings = torch.cat(all_embeddings, dim=0).numpy()
    return dataset, embeddings