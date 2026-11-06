"""Load any checkpoint (original or new) and predict, with optional test-time augmentation.

TTA: predict on flipped copies of the photo and average in real units.
Ensemble: average the predictions of several checkpoints.
"""
import numpy as np
import torch

from config import CHECKPOINT_DIR, IMAGE_SIZE
from data import TargetScaler, build_transforms
from model import MultiTaskResNet

LEGACY_CONFIG = dict(backbone="resnet18", head="direct", img_size=IMAGE_SIZE, keep_aspect=False, strong_aug=False)


class Predictor:
    def __init__(self, name: str, device=None):
        self.name = name
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        ckpt = torch.load(CHECKPOINT_DIR / f"{name}.pth", map_location="cpu", weights_only=True)
        self.config = {**LEGACY_CONFIG, **ckpt.get("model_config", {})}
        self.scaler = TargetScaler.from_checkpoint(ckpt)
        self.model = MultiTaskResNet(pretrained=False, backbone=self.config["backbone"], head=self.config["head"])
        self.model.load_state_dict(ckpt["model_state_dict"], strict=True)
        self.model.to(self.device).eval()
        self.transform = build_transforms(False, img_size=self.config["img_size"], keep_aspect=self.config["keep_aspect"])

    def views(self, x, tta: bool):
        if not tta:
            return [x]
        views = [x, torch.flip(x, dims=[3])]  # horizontal flip: seen in every training run
        if self.config["strong_aug"]:  # vertical flips only if the model was trained with them
            views += [torch.flip(x, dims=[2]), torch.flip(x, dims=[2, 3])]
        return views

    @torch.inference_mode()
    def predict_batch(self, images: torch.Tensor, tta: bool = False) -> np.ndarray:
        """images: normalised tensor batch -> (n, 5) predictions in real units, clipped at 0."""
        images = images.to(self.device)
        preds = [self.scaler.inverse(self.model(v).float().cpu().numpy()) for v in self.views(images, tta)]
        return np.clip(np.mean(preds, axis=0), 0, None)

    def predict_image(self, pil_image, tta: bool = True) -> np.ndarray:
        return self.predict_batch(self.transform(pil_image).unsqueeze(0), tta)[0]


class Ensemble:
    """Average of several Predictors (each may have its own image size)."""

    def __init__(self, names, device=None):
        self.members = [Predictor(n, device) for n in names]
        self.name = "ensemble(" + "+".join(names) + ")"

    def predict_image(self, pil_image, tta: bool = True) -> np.ndarray:
        return np.mean([m.predict_image(pil_image, tta) for m in self.members], axis=0)
