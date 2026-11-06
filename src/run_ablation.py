"""Run the full ablation, one improvement at a time, then evaluate everything.

    python src/run_ablation.py            # all runs (skips runs whose checkpoint exists)
    python src/run_ablation.py --quick    # 1 epoch each: checks that every option runs
    python src/run_ablation.py --only R2 R3

Each run adds ONE change to the previous one, so the results table shows how
much each change helps. Expect ~15-25 min per run on a recent GPU.
"""
import argparse
import subprocess
import sys

import pandas as pd

from config import CHECKPOINT_DIR, RESULTS_DIR

COMMON = ["--epochs", "30", "--finetune", "full"]
RUNS = {
    # name: (what changes, extra flags; each row includes the previous row's flags)
    "R1_recipe": ("full fine-tuning + cosine schedule, 30 epochs", []),
    "R2_log": ("+ log targets", ["--log-targets"]),
    "R3_res384": ("+ 384px, 4:3 aspect kept", ["--log-targets", "--img-size", "384", "--keep-aspect"]),
    "R4_aug": ("+ vertical flips, colour jitter", ["--log-targets", "--img-size", "384", "--keep-aspect", "--strong-aug"]),
    "R5_physics": ("+ mass x density head, Atwater loss",
                   ["--log-targets", "--img-size", "384", "--keep-aspect", "--strong-aug",
                    "--head", "density", "--atwater-weight", "0.1"]),
    "R6_convnext": ("R5 with ConvNeXt-Tiny backbone",
                    ["--log-targets", "--img-size", "384", "--keep-aspect", "--strong-aug",
                     "--head", "density", "--atwater-weight", "0.1", "--backbone", "convnext_tiny",
                     "--lr-backbone", "5e-5", "--batch-size", "16"]),
}


def run(cmd):
    print("\n>>>", " ".join(cmd), flush=True)
    subprocess.run([sys.executable, *cmd], check=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--only", nargs="*", help="run ids, e.g. R2 R3")
    p.add_argument("--quick", action="store_true", help="1 epoch per run, names prefixed with quick_")
    args = p.parse_args()

    names = []
    for name, (desc, flags) in RUNS.items():
        if args.only and name.split("_")[0] not in args.only:
            continue
        run_name = f"quick_{name}" if args.quick else name
        names.append(run_name)
        if (CHECKPOINT_DIR / f"{run_name}.pth").exists() and not args.quick:
            print(f"skip {run_name} (checkpoint exists)")
            continue
        print(f"\n=== {run_name}: {desc}")
        common = ["--epochs", "1", "--finetune", "full"] if args.quick else COMMON
        run(["src/train.py", "--name", run_name, *common, *flags])

    run(["src/evaluate.py", "--models", *names, "--tta"])
    if args.quick:  # smoke test only: remove its checkpoints and prediction rows
        for name in names:
            (CHECKPOINT_DIR / f"{name}.pth").unlink(missing_ok=True)
        for split in ("val", "test"):
            path = RESULTS_DIR / "evaluation" / f"predictions_{split}.csv"
            df = pd.read_csv(path)
            df[~df["model"].str.startswith("quick_")].to_csv(path, index=False)
        print("\nQuick check passed: every option trains and evaluates. Now run without --quick.")
        return
    run(["src/report.py"])


if __name__ == "__main__":
    main()
