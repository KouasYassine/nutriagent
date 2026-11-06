"""PyTorch dataset, image transforms and target normalisation."""
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

from config import IMAGE_SIZE, IMAGENET_MEAN, IMAGENET_STD, TARGETS


def build_transforms(train: bool):
    steps = [transforms.Resize((IMAGE_SIZE, IMAGE_SIZE))]
    if train:
        # A flip does not change the food on the plate, so labels stay valid.
        steps += [transforms.RandomHorizontalFlip(p=0.5)]
    steps += [transforms.ToTensor(), transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD)]
    return transforms.Compose(steps)


class TargetScaler:
    """z-score normalisation fitted on the training split only."""

    def __init__(self, mean: dict, std: dict):
        self.mean = np.array([mean[t] for t in TARGETS], dtype=np.float32)
        self.std = np.array([std[t] for t in TARGETS], dtype=np.float32)

    @classmethod
    def fit(cls, train_df):
        return cls(train_df[TARGETS].mean().to_dict(), train_df[TARGETS].std().to_dict())

    @classmethod
    def from_checkpoint(cls, ckpt):
        if list(ckpt["target_cols"]) != TARGETS:
            raise ValueError(f"Checkpoint target order {ckpt['target_cols']} != {TARGETS}")
        return cls(ckpt["target_mean"], ckpt["target_std"])

    def to_dicts(self):
        return ({t: float(m) for t, m in zip(TARGETS, self.mean)},
                {t: float(s) for t, s in zip(TARGETS, self.std)})

    def transform(self, values: np.ndarray) -> np.ndarray:
        return (values - self.mean) / self.std

    def inverse(self, values: np.ndarray) -> np.ndarray:
        return values * self.std + self.mean


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
