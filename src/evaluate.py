"""Run every checkpoint on the validation and test splits and save raw predictions.

    python src/evaluate.py                    # all checkpoints in checkpoints/
    python src/evaluate.py --models A2_Mass2  # just one

Writes results/evaluation/predictions_<split>.csv (one row per model x dish).
Metrics, figures and uncertainty intervals are then computed from these files
by `src/report.py` and `src/uncertainty.py`, which do not need a GPU.
"""
import argparse

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from config import CHECKPOINT_DIR, RESULTS_DIR, TARGETS, load_labels, make_splits
from data import NutritionDataset, TargetScaler, build_transforms
from model import MultiTaskResNet


@torch.inference_mode()
def predict(model, df, scaler, device, batch_size=32):
    loader = DataLoader(NutritionDataset(df, scaler, build_transforms(False)), batch_size=batch_size)
    outputs = [model(images.to(device)).cpu().numpy() for images, _ in loader]
    # Nutrient amounts cannot be negative.
    return np.clip(scaler.inverse(np.concatenate(outputs)), 0.0, None)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--models", nargs="*", help="checkpoint names without .pth (default: all)")
    args = p.parse_args()

    names = args.models or sorted(c.stem for c in CHECKPOINT_DIR.glob("*.pth"))
    if not names:
        raise SystemExit(f"No checkpoints found in {CHECKPOINT_DIR}")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    _, val_df, test_df = make_splits(load_labels())
    out_dir = RESULTS_DIR / "evaluation"
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = {"val": [], "test": []}

    for name in names:
        ckpt = torch.load(CHECKPOINT_DIR / f"{name}.pth", map_location="cpu", weights_only=True)
        scaler = TargetScaler.from_checkpoint(ckpt)
        model = MultiTaskResNet(pretrained=False)
        model.load_state_dict(ckpt["model_state_dict"], strict=True)
        model.to(device).eval()

        for split, df in [("val", val_df), ("test", test_df)]:
            pred = predict(model, df, scaler, device)
            out = pd.DataFrame({"model": name, "dish_id": df["dish_id"].to_numpy()})
            for i, t in enumerate(TARGETS):
                out[f"{t}_actual"] = df[t].to_numpy()
                out[f"{t}_predicted"] = pred[:, i]
            rows[split].append(out)
            mae = np.abs(pred - df[TARGETS].to_numpy()).mean(axis=0)
            print(f"{name:10s} {split:4s} MAE | " + " ".join(f"{t} {m:.2f}" for t, m in zip(TARGETS, mae)))

    for split, frames in rows.items():
        path = out_dir / f"predictions_{split}.csv"
        pd.concat(frames).to_csv(path, index=False)
        print("Saved", path)
    print("Next: python src/report.py && python src/uncertainty.py")


if __name__ == "__main__":
    main()
