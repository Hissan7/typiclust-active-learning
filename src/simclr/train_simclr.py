import os
import torch
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToPILImage

from simclr.augmentations import SimCLRTransform
from simclr.simclr_model import SimCLR
from simclr.contrastive_loss import nt_xent_loss


class CIFAR10PairDataset:
    def __init__(self, root="./data", train=True, download=True, transform=None):
        self.dataset = datasets.CIFAR10(root=root, train=train, download=download)
        self.transform = transform
        self.to_pil = ToPILImage()

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        image, _ = self.dataset[idx]

        if self.transform is None:
            raise ValueError("Transform must not be None for SimCLR training.")

        x1, x2 = self.transform(image)
        return x1, x2


def get_device():
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def train_simclr(
    epochs: int = 50,
    batch_size: int = 256,
    lr: float = 1e-3,
    temperature: float = 0.5,
    save_path: str = "models/simclr_resnet18.pth"
):
    device = get_device()
    print("SimCLR training device:", device)

    transform = SimCLRTransform(image_size=32)
    dataset = CIFAR10PairDataset(transform=transform)

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0
    )

    model = SimCLR(projection_dim=128).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    model.train()
    for epoch in range(epochs):
        running_loss = 0.0

        for x1, x2 in loader:
            x1 = x1.to(device)
            x2 = x2.to(device)

            _, z1 = model(x1)
            _, z2 = model(x2)

            loss = nt_xent_loss(z1, z2, temperature=temperature)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        avg_loss = running_loss / len(loader)
        print(f"Epoch {epoch + 1}/{epochs} | SimCLR Loss: {avg_loss:.4f}")

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    torch.save(model.state_dict(), save_path)
    print(f"Saved SimCLR model to {save_path}")

    return model