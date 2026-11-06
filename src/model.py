import torch
import torch.nn as nn
from torchvision import models

from config import TARGETS

# torchvision backbones: (constructor, ImageNet weights, feature dim, name of last block)
BACKBONES = {
    "resnet18": (models.resnet18, models.ResNet18_Weights.DEFAULT, 512, "layer4"),
    "resnet50": (models.resnet50, models.ResNet50_Weights.DEFAULT, 2048, "layer4"),
    "convnext_tiny": (models.convnext_tiny, models.ConvNeXt_Tiny_Weights.DEFAULT, 768, "features.7"),
    "efficientnet_v2_s": (models.efficientnet_v2_s, models.EfficientNet_V2_S_Weights.DEFAULT, 1280, "features.6"),
}


def build_backbone(name: str, pretrained: bool):
    ctor, weights, dim, _ = BACKBONES[name]
    net = ctor(weights=weights if pretrained else None)
    if name.startswith("resnet"):
        net.fc = nn.Identity()
    else:  # convnext / efficientnet: keep pooling + norm, drop the final Linear
        net.classifier[-1] = nn.Identity()
    return net, dim


class MultiTaskResNet(nn.Module):
    """Backbone -> shared 256-d layer -> nutrient outputs (in normalised target space).

    head="direct"  : one linear head per nutrient (original experiments).
    head="density" : physics-structured head. The network predicts the portion mass
                     and the nutrient content PER GRAM; totals = mass x density.
                     This separates "how much food" from "what food" and keeps the
                     5 outputs consistent with each other. Requires log targets.

    Attribute names are unchanged for the default config, so the original
    checkpoints (A1, A1_Phase2, A2_Mass2) load with strict=True.
    """

    def __init__(self, pretrained: bool = True, dropout: float = 0.2, backbone: str = "resnet18",
                 head: str = "direct"):
        super().__init__()
        self.backbone_name, self.head_type = backbone, head
        self.backbone, n_features = build_backbone(backbone, pretrained)
        self.shared = nn.Sequential(nn.Linear(n_features, 256), nn.ReLU(), nn.Dropout(dropout))
        self.calories_head = nn.Linear(256, 1)
        self.mass_head = nn.Linear(256, 1)
        self.fat_head = nn.Linear(256, 1)
        self.carbs_head = nn.Linear(256, 1)
        self.protein_head = nn.Linear(256, 1)
        if head == "density":
            # log1p-target normalisation stats, copied in by set_target_stats()
            self.register_buffer("t_mean", torch.zeros(len(TARGETS)))
            self.register_buffer("t_std", torch.ones(len(TARGETS)))

    def set_target_stats(self, mean, std):
        self.t_mean.copy_(torch.as_tensor(mean, dtype=torch.float32))
        self.t_std.copy_(torch.as_tensor(std, dtype=torch.float32))

    def last_block(self):
        module = self.backbone
        for part in BACKBONES[self.backbone_name][3].split("."):
            module = module[int(part)] if part.isdigit() else getattr(module, part)
        return module

    def heads(self):
        return [getattr(self, f"{t}_head") for t in TARGETS]

    def head_parameters(self):
        params = list(self.shared.parameters())
        for head in self.heads():
            params += list(head.parameters())
        return params

    def forward(self, x):
        h = self.shared(self.backbone(x))
        raw = torch.cat([head(h) for head in self.heads()], dim=1)
        if self.head_type == "direct":
            return raw

        # density head: mass_head -> log grams; other heads -> log(amount per gram).
        # Computed in float32: exp() of fp16 values overflows above ~11.
        raw = raw.float()
        i_mass = TARGETS.index("mass")
        log_mass = raw[:, i_mass:i_mass + 1] + 5.0           # centred near ~150 g
        log_density = raw - 1.0                               # centred near ~0.4 per g
        is_mass = torch.zeros_like(raw, dtype=torch.bool)
        is_mass[:, i_mass] = True
        log_total = torch.where(is_mass, log_mass.expand_as(raw), log_mass + log_density).clamp(-12, 12)
        totals = torch.exp(log_total)
        return (torch.log1p(totals) - self.t_mean) / self.t_std
