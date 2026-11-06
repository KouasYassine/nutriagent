"""Resize every Nutrition5k photo once and save it as a JPEG, for fast training.

    python src/cache_images.py              # short side 384 px -> 384 x 512 (keeps 4:3)

Decoding a 640x480 PNG is the training bottleneck (the GPU waits for the CPU).
A 384x512 JPEG (quality 95) decodes several times faster and is visually identical
for this task. Then add --cache to train.py / evaluate.py (run_ablation.py does it).
"""
import argparse
from concurrent.futures import ThreadPoolExecutor

from PIL import Image

from config import CACHE_DIR, cache_path, load_labels


def resize_one(args):
    src, dst, short_side = args
    if dst.exists():
        return 0
    img = Image.open(src).convert("RGB")
    w, h = img.size
    scale = short_side / min(w, h)
    img = img.resize((round(w * scale), round(h * scale)), Image.BICUBIC)
    img.save(dst, "JPEG", quality=95)
    return 1


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--short-side", type=int, default=384)
    p.add_argument("--workers", type=int, default=8)
    args = p.parse_args()

    df = load_labels()
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    jobs = [(src, cache_path(d), args.short_side) for src, d in zip(df["image_path"], df["dish_id"])]
    with ThreadPoolExecutor(args.workers) as pool:
        done = sum(pool.map(resize_one, jobs))
    print(f"Resized {done} images ({len(jobs) - done} already cached) into {CACHE_DIR}")


if __name__ == "__main__":
    main()
