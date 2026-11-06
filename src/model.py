import torch
import torch.nn as nn
from torchvision import models

from config import TARGETS


class MultiTaskResNet(nn.Module):
    """ResNet18 backbone -> shared 256-d layer -> one linear head per nutrient.

    Attribute names are kept identical to the original experiments so older
    checkpoints (A1, A1_Phase2, A2_Mass2) load with strict=True.
    """

    def __init__(self, pretrained: bool = True, dropout: float = 0.2):
        super().__init__()
        backbone = models.resnet18(weights=models.ResNet18_Weights.DEFAULT if pretrained else None)
        n_features = backbone.fc.in_features
        backbone.fc = nn.Identity()
        self.backbone = backbone

        self.shared = nn.Sequential(nn.Linear(n_features, 256), nn.ReLU(), nn.Dropout(dropout))
        self.calories_head = nn.Linear(256, 1)
        self.mass_head = nn.Linear(256, 1)
        self.fat_head = nn.Linear(256, 1)
        self.carbs_head = nn.Linear(256, 1)
        self.protein_head = nn.Linear(256, 1)

    def heads(self):
        return [getattr(self, f"{t}_head") for t in TARGETS]

    def head_parameters(self):
        params = list(self.shared.parameters())
        for head in self.heads():
            params += list(head.parameters())
        return params

    def forward(self, x):
        h = self.shared(self.backbone(x))
        return torch.cat([head(h) for head in self.heads()], dim=1)
