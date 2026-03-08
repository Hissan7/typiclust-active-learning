import torch
import torchvision.models as models


print("hello")
class FeatureExtractor:
    def __init__(self, device: str = None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        print("Using device:", self.device)

        weights = models.ResNet18_Weights.DEFAULT
        model = models.resnet18(weights=weights)

        self.model = torch.nn.Sequential(*list(model.children())[:-1])
        self.model.to(self.device)
        self.model.eval()

    def extract_features(self, dataloader):
        all_features = []

        with torch.no_grad():
            for images, _ in dataloader:
                images = images.to(self.device)
                features = self.model(images)
                features = features.view(features.size(0), -1)
                all_features.append(features.cpu())

        return torch.cat(all_features, dim=0)