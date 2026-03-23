import torch
import torch.nn as nn
import torchvision.models as models


class SimCLR(nn.Module):
    def __init__(self, projection_dim: int = 128):
        super().__init__()

        backbone = models.resnet18(weights=None)

        self.encoder = nn.Sequential(*list(backbone.children())[:-1])
        feature_dim = backbone.fc.in_features  # 512 for ResNet18

        self.projector = nn.Sequential(
            nn.Linear(feature_dim, feature_dim),
            nn.ReLU(),
            nn.Linear(feature_dim, projection_dim)
        )

    def forward(self, x):
        h = self.encoder(x)              
        h = torch.flatten(h, start_dim=1) 
        z = self.projector(h)            
        return h, z