"""Run checkpoints on the validation and test splits and save raw predictions.

    python src/evaluate.py                                  # every checkpoint in checkpoints/
    python src/evaluate.py --models v3_full --tta           # one model, with flip TTA
    python src/evaluate.py --models v3_s1 v3_s2 v3_s3 --tta --ensemble v3_ens

Writes results/evaluation/predictions_<split>.csv (one row per model x dish).
Rows for models evaluated again are replaced; other models are kept, so the
files accumulate the full ablation. Then run src/report.py and src/uncertainty.py.
"""
import argparse
import os

import numpy as np
import pandas as pd
from torch.utils.data import DataLoader

from config import CHECKPOINT_DIR, RESULTS_DIR, TARGETS, load_labels, make_splits
from data import NutritionDataset
from inference import Predictor

OUT_DIR = RESULTS_DIR / "evaluation"


def predict_split(pred: Predictor, df, tta, batch_size):
    ds = NutritionDataset(df, pred.scaler, pred.transform)
    loader = DataLoader(ds, batch_size=batch_size, num_workers=0 if os.name == "nt" else 2)
    return np.concatenate([pred.predict_batch(images, tta) for images, _ in loader])


def to_frame(label, df, pred):
    out = pd.DataFrame({"model": label, "dish_id": df["dish_id"].to_numpy()})
    for i, t in enumerate(TARGETS):
        out[f"{t}_actual"] = df[t].to_numpy()
        out[f"{t}_predicted"] = pred[:, i]
    return out


def save(split, frames):
    path = OUT_DIR / f"predictions_{split}.csv"
    new = pd.concat(frames)
    if path.exists():
        old = pd.read_csv(path)
        new = pd.concat([old[~old["model"].isin(new["model"].unique())], new])
    new.to_csv(path, index=False)
    print("Saved", path)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--models", nargs="*", help="checkpoint names without .pth (default: all)")
    p.add_argument("--tta", action="store_true", help="average predictions over flipped images")
    p.add_argument("--ensemble", help="also save the average of --models under this name")
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--cache", action="store_true", help="read pre-resized JPEGs")
    args = p.parse_args()

    names = args.models or sorted(c.stem for c in CHECKPOINT_DIR.glob("*.pth"))
    if not names:
        raise SystemExit(f"No checkpoints found in {CHECKPOINT_DIR}")
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    _, val_df, test_df = make_splits(load_labels(use_cache=args.cache))
    frames, member_preds = {"val": [], "test": []}, {"val": [], "test": []}

    for name in names:
        pred = Predictor(name)
        label = name + ("+tta" if args.tta else "")
        for split, df in [("val", val_df), ("test", test_df)]:
            y_hat = predict_split(pred, df, args.tta, args.batch_size)
            member_preds[split].append(y_hat)
            frames[split].append(to_frame(label, df, y_hat))
            mae = np.abs(y_hat - df[TARGETS].to_numpy()).mean(axis=0)
            print(f"{label:22s} {split:4s} MAE | " + " ".join(f"{t} {m:.2f}" for t, m in zip(TARGETS, mae)))

    if args.ensemble:
        for split, df in [("val", val_df), ("test", test_df)]:
            y_hat = np.mean(member_preds[split], axis=0)
            frames[split].append(to_frame(args.ensemble, df, y_hat))
            mae = np.abs(y_hat - df[TARGETS].to_numpy()).mean(axis=0)
            print(f"{args.ensemble:22s} {split:4s} MAE | " + " ".join(f"{t} {m:.2f}" for t, m in zip(TARGETS, mae)))
        (OUT_DIR / f"{args.ensemble}.members.txt").write_text("\n".join(names) + "\n")

    for split in ("val", "test"):
        save(split, frames[split])
    print("Next: python src/report.py && python src/uncertainty.py --model <best model>")


if __name__ == "__main__":
    main()
