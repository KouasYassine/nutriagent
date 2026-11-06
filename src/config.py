"""Single source of truth for paths, targets and the train/val/test split."""
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parent.parent

DATA_DIR = ROOT / "data"
RAW_DIR = DATA_DIR / "raw" / "nutrition5k"
LABELS_CSV = DATA_DIR / "processed" / "nutrition5k_labels.csv"
CACHE_DIR = DATA_DIR / "cache"  # pre-resized JPEGs, built by src/cache_images.py

CHECKPOINT_DIR = ROOT / "checkpoints"
RESULTS_DIR = ROOT / "results"

# Order matters: every model output and checkpoint uses this order.
TARGETS = ["calories", "mass", "fat", "carbs", "protein"]
UNITS = {"calories": "kcal", "mass": "g", "fat": "g", "carbs": "g", "protein": "g"}

SEED = 42
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]
IMAGE_SIZE = 224


def resolve_image_path(path) -> str:
    """Accept Windows or POSIX relative paths and resolve them from the repo root."""
    p = Path(str(path).replace("\\", "/"))
    return str(p if p.is_absolute() else ROOT / p)


def cache_path(dish_id: str) -> Path:
    return CACHE_DIR / f"{dish_id}.jpg"


def load_labels(csv_path: Path = LABELS_CSV, use_cache: bool = False) -> pd.DataFrame:
    """use_cache=True reads the pre-resized JPEGs (much faster to decode than the
    original 640x480 PNGs). The split does not change: it only depends on the rows."""
    df = pd.read_csv(csv_path)
    df["image_path"] = df["image_path"].map(resolve_image_path)
    if use_cache:
        cached = df["dish_id"].map(lambda d: str(cache_path(d)))
        missing = int((~cached.map(lambda p: Path(p).exists())).sum())
        if missing:
            raise FileNotFoundError(f"{missing} cached images missing. Run: python src/cache_images.py")
        df["image_path"] = cached
    return df


def make_splits(df: pd.DataFrame, seed: int = SEED):
    """80 / 10 / 10 random split by dish (each dish has exactly one image).

    Kept identical to the split used for all reported experiments, so results
    stay comparable across runs.
    """
    train_df, temp_df = train_test_split(df, test_size=0.20, random_state=seed)
    val_df, test_df = train_test_split(temp_df, test_size=0.50, random_state=seed)
    return train_df.copy(), val_df.copy(), test_df.copy()
