"""Build data/processed/nutrition5k_labels.csv from the raw Nutrition5k download.

Expected layout (Kaggle mirror or official release):
    data/raw/nutrition5k/metadata/dish_metadata_cafe1.csv
    data/raw/nutrition5k/metadata/dish_metadata_cafe2.csv
    data/raw/nutrition5k/realsense_overhead/<dish_id>/rgb.png

Run from the repo root:  python src/prepare_data.py
"""
import csv

import pandas as pd

from config import LABELS_CSV, RAW_DIR, ROOT


def read_dish_metadata(metadata_dir):
    """Dish-level totals are the first 6 fields of each metadata row:
    dish_id, total_calories, total_mass, total_fat, total_carb, total_protein.
    The remaining fields are per-ingredient details, which we do not use.
    """
    rows = []
    for file in sorted(metadata_dir.glob("dish_metadata_cafe*.csv")):
        with open(file, "r", encoding="utf-8") as f:
            for row in csv.reader(f):
                if len(row) < 6:
                    continue
                rows.append({
                    "dish_id": row[0],
                    "calories": float(row[1]),
                    "mass": float(row[2]),
                    "fat": float(row[3]),
                    "carbs": float(row[4]),
                    "protein": float(row[5]),
                })
    return pd.DataFrame(rows).drop_duplicates("dish_id")


def main():
    labels = read_dish_metadata(RAW_DIR / "metadata")
    print("Dishes with labels:", len(labels))

    image_paths = {p.parent.name: p for p in (RAW_DIR / "realsense_overhead").glob("*/rgb.png")}
    print("Overhead RGB images found:", len(image_paths))

    labels["image_path"] = labels["dish_id"].map(image_paths)
    labels = labels[labels["image_path"].notna()].copy()
    # Store POSIX paths relative to the repo root so the CSV works on any OS.
    labels["image_path"] = labels["image_path"].map(lambda p: p.relative_to(ROOT).as_posix())
    print("Usable dishes (label + image):", len(labels))

    LABELS_CSV.parent.mkdir(parents=True, exist_ok=True)
    labels.to_csv(LABELS_CSV, index=False)
    print("Saved", LABELS_CSV)


if __name__ == "__main__":
    main()
