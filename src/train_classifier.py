import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.models as models
from torch.utils.data import DataLoader, Subset


def get_device():
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def create_selected_subset(dataset, selected_indices):
    return Subset(dataset, selected_indices)


def create_model(num_classes: int = 10):
    model = models.resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model


def train_model(train_dataset, test_loader, epochs: int = 5, batch_size: int = 32):
    device = get_device()
    print("Training device:", device)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    model = create_model(num_classes=10).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss = running_loss + loss.item()
            _, predicted = torch.max(outputs, 1)
            total = total + labels.size(0)
            correct = correct + (predicted == labels).sum().item()

        train_acc = 100.0 * correct / total if total > 0 else 0.0
        print(
            f"Epoch {epoch + 1}/{epochs} | "
            f"Loss: {running_loss:.4f} | "
            f"Train Acc: {train_acc:.2f}%"
        )

    test_acc = evaluate_model(model, test_loader, device)
    return model, test_acc


def evaluate_model(model, test_loader, device):
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)

            outputs = model(images)
            _, predicted = torch.max(outputs, 1)

            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = 100.0 * correct / total if total > 0 else 0.0
    print(f"Test Accuracy: {accuracy:.2f}%")
    return accuracy