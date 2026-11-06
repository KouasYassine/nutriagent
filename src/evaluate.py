
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from PIL import Image
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from torchvision import transforms

from dataset import NutritionDataset
from model import MultiTaskResNet


# =====================================================
# CONFIGURATION
# =====================================================

ROOT = Path(__file__).resolve().parent.parent
TARGETS = ["calories", "mass", "fat", "carbs", "protein"]

CHECKPOINTS = {
    "A1": "best_multitask_resnet18_A1.pth",
    "A1_Phase2": "best_multitask_resnet18_A1_phase2.pth",
    "A2_Mass2": "best_multitask_resnet18_A2_mass2.pth",
}

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

OUTPUT_DIR = ROOT / "results" / "evaluation"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

print("Device:", DEVICE)


# =====================================================
# LOAD TEST DATA
# =====================================================

df = pd.read_csv(
    ROOT / "data/processed/nutrition5k_labels.csv"
)

# Reproduce the original training split.
train_df, temp_df = train_test_split(
    df, test_size=0.20, random_state=42
)

val_df, test_df = train_test_split(
    temp_df, test_size=0.50, random_state=42
)

test_df = test_df.copy()

# Resolve image paths relative to project root.
def resolve_image_path(path):
    return str(ROOT / Path(str(path).replace("\\", "/")))

test_df["image_path"] = test_df["image_path"].apply(
    resolve_image_path
)

missing = [
    p for p in test_df["image_path"]
    if not Path(p).is_file()
]

if missing:
    raise FileNotFoundError(
        f"{len(missing)} test images are missing. "
        f"First missing image: {missing[0]}"
    )

print("Test samples:", len(test_df))


# =====================================================
# IMAGE TRANSFORM
# =====================================================

eval_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])


# =====================================================
# EVALUATE CHECKPOINTS
# =====================================================

results = []
predictions = []

for model_name, filename in CHECKPOINTS.items():

    print(f"\nEvaluating {model_name}...")

    checkpoint = torch.load(
        ROOT / filename,
        map_location="cpu",
        weights_only=True
    )

    if checkpoint["target_cols"] != TARGETS:
        raise ValueError(
            f"Target order mismatch for {model_name}"
        )

    mean = np.array(
        [checkpoint["target_mean"][t] for t in TARGETS],
        dtype=np.float32
    )

    std = np.array(
        [checkpoint["target_std"][t] for t in TARGETS],
        dtype=np.float32
    )

    # Build normalized targets from checkpoint metadata.
    current_df = test_df.copy()

    for i, target in enumerate(TARGETS):
        current_df[f"{target}_scaled"] = (
            current_df[target] - mean[i]
        ) / std[i]

    dataset = NutritionDataset(
        current_df,
        transform=eval_transform
    )

    loader = DataLoader(
        dataset,
        batch_size=16,
        shuffle=False,
        num_workers=0
    )

    model = MultiTaskResNet(pretrained=False).to(DEVICE)

    model.load_state_dict(
        checkpoint["model_state_dict"],
        strict=True
    )

    model.eval()

    y_pred_scaled = []
    y_true_scaled = []

    with torch.inference_mode():
        for images, targets in loader:
            images = images.to(DEVICE)

            outputs = model(images)

            y_pred_scaled.append(outputs.cpu().numpy())
            y_true_scaled.append(targets.numpy())

    y_pred_scaled = np.concatenate(y_pred_scaled)
    y_true_scaled = np.concatenate(y_true_scaled)

    # Convert predictions and targets to original units.
    y_pred = y_pred_scaled * std + mean
    y_true = y_true_scaled * std + mean

    for i, target in enumerate(TARGETS):

        mae = mean_absolute_error(
            y_true[:, i], y_pred[:, i]
        )

        rmse = np.sqrt(
            mean_squared_error(
                y_true[:, i], y_pred[:, i]
            )
        )

        results.append({
            "model": model_name,
            "target": target,
            "MAE": mae,
            "RMSE": rmse
        })

        print(
            f"{target:10s} "
            f"MAE={mae:9.2f} "
            f"RMSE={rmse:9.2f}"
        )

    for row_idx, (_, row) in enumerate(test_df.iterrows()):
        record = {
            "model": model_name,
            "dish_id": row["dish_id"],
        }
        for i, target in enumerate(TARGETS):
            record[f"{target}_actual"] = float(y_true[row_idx, i])
            record[f"{target}_predicted"] = float(y_pred[row_idx, i])
        predictions.append(record)


# =====================================================
# SAVE RESULTS
# =====================================================

results_df = pd.DataFrame(results)

results_df.to_csv(
    OUTPUT_DIR / "model_comparison.csv",
    index=False
)

pd.DataFrame(predictions).to_csv(
    OUTPUT_DIR / "predictions.csv",
    index=False
)

print("\nEvaluation complete!")
print("\nMAE comparison:")
print(
    results_df.pivot(
        index="model",
        columns="target",
        values="MAE"
    ).round(2).to_string()
)

print("\nSaved results to:", OUTPUT_DIR)
