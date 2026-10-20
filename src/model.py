import torch
import torch.nn as nn
from torchvision import models
from torchvision.models import ResNet18_Weights


class CalorieResNet(nn.Module):
    def __init__(self):
        super().__init__()

        self.backbone = models.resnet18(
            weights=ResNet18_Weights.DEFAULT
        )

        self.in_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Linear(
            self.in_features,
            1
        )

    def forward(self, x):
        return self.backbone(x)


class MultiTaskResNet(nn.Module):
    def __init__(self):
        super().__init__()

        backbone = models.resnet18(
            weights=models.ResNet18_Weights.DEFAULT
        )

        n_features = backbone.fc.in_features
        backbone.fc = nn.Identity()

        self.backbone = backbone

        self.shared = nn.Sequential(
            nn.Linear(n_features, 256),
            nn.ReLU(),
            nn.Dropout(0.2)
        )

        self.calories_head = nn.Linear(256, 1)
        self.mass_head = nn.Linear(256, 1)
        self.fat_head = nn.Linear(256, 1)
        self.carbs_head = nn.Linear(256, 1)
        self.protein_head = nn.Linear(256, 1)

    def forward(self, x):
        features = self.backbone(x)
        shared_features = self.shared(features)

        calories = self.calories_head(shared_features)
        mass = self.mass_head(shared_features)
        fat = self.fat_head(shared_features)
        carbs = self.carbs_head(shared_features)
        protein = self.protein_head(shared_features)

        return torch.cat(
            [calories, mass, fat, carbs, protein],
            dim=1
        )
