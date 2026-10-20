from pathlib import Path
import pandas as pd

DATA_ROOT = Path("data/raw/nutrition5k")
IMAGE_DIR = DATA_ROOT / "realsense_overhead"
METADATA_DIR = DATA_ROOT / "metadata"
metadata_files = list(
    METADATA_DIR.glob("dish_metadata_cafe*.csv")
)

frames = []

for file in metadata_files:
   import csv

rows = []

for file in metadata_files:
    with open(file, "r", encoding="utf-8") as f:
        reader = csv.reader(f)

        for row in reader:
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

df = pd.DataFrame(rows)


# -----------------------------
# Find all RGB images
# -----------------------------

image_paths = list(
    IMAGE_DIR.glob("*/rgb.png")
)

print("RGB images found:", len(image_paths))

image_map = {
    path.parent.name: path
    for path in image_paths
}

# -----------------------------
# Match images to labels
# -----------------------------

df["image_path"] = df["dish_id"].map(image_map)

df = df[df["image_path"].notna()].copy()

print("Usable dishes:", len(df))

# -----------------------------
# Save processed dataframe
# -----------------------------

OUTPUT_PATH = Path(
    "data/processed/nutrition5k_labels.csv"
)

OUTPUT_PATH.parent.mkdir(
    parents=True,
    exist_ok=True
)

df.to_csv(
    OUTPUT_PATH,
    index=False
)

print(f"Saved to {OUTPUT_PATH}")
