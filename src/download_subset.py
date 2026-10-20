import os
import subprocess
import pandas as pd

CSV_PATH = "data/raw/dish_nutrition_values.csv"
OUTPUT_DIR = "data/raw/images"

N_IMAGES = 1000

df = pd.read_csv(CSV_PATH)

os.makedirs(OUTPUT_DIR, exist_ok=True)

subset = df.head(N_IMAGES)

successful_rows = []

for i, row in subset.iterrows():
    dish_id = row["dish_id"]

    kaggle_path = (
        f"imagery/realsense_overhead/"
        f"{dish_id}/rgb.png"
    )

    print(f"[{len(successful_rows)}/{N_IMAGES}] Downloading {dish_id}")

    result = subprocess.run(
        [
            "kaggle",
            "datasets",
            "download",
            "gillesokhin/nutrition5k-dataset",
            "-f",
            kaggle_path,
            "-p",
            OUTPUT_DIR,
            "--unzip",
        ]
    )

    if result.returncode == 0:
        successful_rows.append(row)

print(f"\nSuccessfully downloaded {len(successful_rows)} images")

subset_df = pd.DataFrame(successful_rows)

subset_df.to_csv(
    "data/raw/subset_labels.csv",
    index=False
)