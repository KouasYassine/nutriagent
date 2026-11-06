"""PyTorch dataset, image transforms and target normalisation."""
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

from config import IMAGE_SIZE, IMAGENET_MEAN, IMAGENET_STD, TARGETS


def build_transforms(train: bool, img_size: int = IMAGE_SIZE, keep_aspect: bool = False,
                     strong_aug: bool = False):
    """
    keep_aspect=False : resize to img_size x img_size (original experiments; squashes 4:3 photos)
    keep_aspect=True  : resize the SHORT side to img_size, keeping 4:3 (e.g. 384 -> 384 x 512)

    No crops or zooms on purpose: the camera height is fixed in Nutrition5k, so the
    apparent size of the food is the main cue for portion size. Zooming would change
    the image without changing the label.
    """
    size = img_size if keep_aspect else (img_size, img_size)
    steps = [transforms.Resize(size)]
    if train:
        steps.append(transforms.RandomHorizontalFlip(p=0.5))
        if strong_aug:
            # Overhead photo: any flip is still a valid photo of the same plate.
            steps += [transforms.RandomVerticalFlip(p=0.5),
                      transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.02)]
    steps += [transforms.ToTensor(), transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD)]
    return transforms.Compose(steps)


class TargetScaler:
    """z-score normalisation fitted on the training split only.

    With log=True the targets are first mapped with log1p. The nutrient
    distributions are very right-skewed (a few 1,000+ kcal plates), so in raw
    units the loss is dominated by a handful of huge dishes.
    """

    def __init__(self, mean: dict, std: dict, log: bool = False):
        self.mean = np.array([mean[t] for t in TARGETS], dtype=np.float32)
        self.std = np.array([std[t] for t in TARGETS], dtype=np.float32)
        self.log = log

    @classmethod
    def fit(cls, train_df, log: bool = False):
        values = train_df[TARGETS].astype(np.float32)
        if log:
            values = np.log1p(values)
        return cls(values.mean().to_dict(), values.std().to_dict(), log=log)

    @classmethod
    def from_checkpoint(cls, ckpt):
        if list(ckpt["target_cols"]) != TARGETS:
            raise ValueError(f"Checkpoint target order {ckpt['target_cols']} != {TARGETS}")
        return cls(ckpt["target_mean"], ckpt["target_std"], log=ckpt.get("log_targets", False))

    def to_dicts(self):
        return ({t: float(m) for t, m in zip(TARGETS, self.mean)},
                {t: float(s) for t, s in zip(TARGETS, self.std)})

    def transform(self, values: np.ndarray) -> np.ndarray:
        values = np.log1p(values) if self.log else values
        return (values - self.mean) / self.std

    def inverse(self, values: np.ndarray) -> np.ndarray:
        out = values * self.std + self.mean
        return np.expm1(out) if self.log else out

    def inverse_torch(self, values: torch.Tensor) -> torch.Tensor:
        """Same as inverse(), differentiable (used by the Atwater consistency loss)."""
        mean = torch.as_tensor(self.mean, device=values.device)
        std = torch.as_tensor(self.std, device=values.device)
        out = values * std + mean
        return torch.expm1(out.clamp(max=12)) if self.log else out


class NutritionDataset(Dataset):
    """Returns (image_tensor, normalised_target[5]) in TARGETS order."""

    def __init__(self, df, scaler: TargetScaler, transform=None):
        self.paths = df["image_path"].tolist()
        self.targets = scaler.transform(df[TARGETS].to_numpy(dtype=np.float32)).astype(np.float32)
        self.transform = transform

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        image = Image.open(self.paths[idx]).convert("RGB")
        if self.transform is not None:
            image = self.transform(image)
        return image, torch.from_numpy(self.targets[idx])
