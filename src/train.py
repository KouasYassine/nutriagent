"""Train the multitask ResNet18 in the three stages used in the experiments.

    A1         ImageNet ResNet18, backbone frozen, train the shared layer + 5 heads
    A1_Phase2  start from A1, also fine-tune backbone.layer4
    A2_Mass2   start from A1_Phase2, same as Phase2 but mass loss weighted x2

Run from the repo root:
    python src/train.py --stage A1
    python src/train.py --stage A1_Phase2
    python src/train.py --stage A2_Mass2

Checkpoints go to checkpoints/<stage>.pth with the target normalisation stats,
so evaluation and the app never need to recompute them.
"""
import argparse
import json
import random
import time

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from config import CHECKPOINT_DIR, RESULTS_DIR, SEED, TARGETS, load_labels, make_splits
from data import NutritionDataset, TargetScaler, build_transforms
from model import MultiTaskResNet

# Hyper-parameters per stage. A2_Mass2 matches the original A2 script exactly.
# A1 / A1_Phase2: check these against your original A1 code and edit if needed.
STAGES = {
    "A1": dict(init=None, unfreeze_layer4=False, epochs=15,
               lr_backbone=0.0, lr_head=1e-3, mass_weight=1.0),
    "A1_Phase2": dict(init="A1", unfreeze_layer4=True, epochs=10,
                      lr_backbone=1e-5, lr_head=1e-4, mass_weight=1.0),
    "A2_Mass2": dict(init="A1_Phase2", unfreeze_layer4=True, epochs=10,
                     lr_backbone=5e-6, lr_head=5e-5, mass_weight=2.0),
}


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def loss_weights(mass_weight, device):
    w = torch.ones(len(TARGETS), device=device)
    w[TARGETS.index("mass")] = mass_weight
    return w


def run_epoch(model, loader, criterion, weights, device, scaler, optimizer=None, unfreeze_layer4=False):
    """One pass over `loader`. Trains if an optimizer is given, otherwise evaluates.
    Returns (mean weighted loss, per-target MAE in real units)."""
    training = optimizer is not None
    model.train(training)
    if training:
        # Frozen backbone blocks keep their ImageNet BatchNorm statistics.
        model.backbone.eval()
        if unfreeze_layer4:
            model.backbone.layer4.train()

    total_loss, abs_err, n = 0.0, np.zeros(len(TARGETS)), 0
    with torch.set_grad_enabled(training):
        for images, targets in loader:
            images, targets = images.to(device), targets.to(device)
            outputs = model(images)
            per_target = torch.stack([criterion(outputs[:, i], targets[:, i]) for i in range(len(TARGETS))])
            loss = (weights * per_target).sum()
            if training:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
            total_loss += loss.item() * images.size(0)
            pred = scaler.inverse(outputs.detach().cpu().numpy())
            true = scaler.inverse(targets.cpu().numpy())
            abs_err += np.abs(pred - true).sum(axis=0)
            n += images.size(0)
    return total_loss / n, abs_err / n


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--stage", choices=STAGES, required=True)
    p.add_argument("--epochs", type=int, help="override the stage default")
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--num-workers", type=int, default=2)
    p.add_argument("--seed", type=int, default=SEED)
    args = p.parse_args()

    cfg = dict(STAGES[args.stage])
    if args.epochs:
        cfg["epochs"] = args.epochs
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Stage {args.stage} | device {device} | config {cfg}")

    train_df, val_df, _ = make_splits(load_labels())  # the test split is never touched here
    print(f"Train {len(train_df)} | Val {len(val_df)}")

    model = MultiTaskResNet(pretrained=cfg["init"] is None).to(device)
    if cfg["init"] is None:
        scaler = TargetScaler.fit(train_df)
    else:
        ckpt = torch.load(CHECKPOINT_DIR / f"{cfg['init']}.pth", map_location="cpu", weights_only=True)
        model.load_state_dict(ckpt["model_state_dict"], strict=True)
        scaler = TargetScaler.from_checkpoint(ckpt)  # keep the same normalisation across stages

    for prm in model.backbone.parameters():
        prm.requires_grad = False
    groups = [{"params": model.head_parameters(), "lr": cfg["lr_head"]}]
    if cfg["unfreeze_layer4"]:
        for prm in model.backbone.layer4.parameters():
            prm.requires_grad = True
        groups.append({"params": model.backbone.layer4.parameters(), "lr": cfg["lr_backbone"]})
    optimizer = torch.optim.AdamW(groups, weight_decay=1e-4)

    loader_kw = dict(batch_size=args.batch_size, num_workers=args.num_workers, pin_memory=device.type == "cuda")
    train_loader = DataLoader(NutritionDataset(train_df, scaler, build_transforms(True)), shuffle=True, **loader_kw)
    val_loader = DataLoader(NutritionDataset(val_df, scaler, build_transforms(False)), shuffle=False, **loader_kw)

    criterion = nn.SmoothL1Loss()
    weights = loss_weights(cfg["mass_weight"], device)
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = CHECKPOINT_DIR / f"{args.stage}.pth"
    best, history = float("inf"), []

    for epoch in range(1, cfg["epochs"] + 1):
        t0 = time.time()
        tr_loss, tr_mae = run_epoch(model, train_loader, criterion, weights, device, scaler,
                                    optimizer, cfg["unfreeze_layer4"])
        va_loss, va_mae = run_epoch(model, val_loader, criterion, weights, device, scaler)
        history.append({"epoch": epoch, "train_loss": tr_loss, "val_loss": va_loss,
                        **{f"train_mae_{t}": float(v) for t, v in zip(TARGETS, tr_mae)},
                        **{f"val_mae_{t}": float(v) for t, v in zip(TARGETS, va_mae)}})
        print(f"[{epoch:02d}/{cfg['epochs']}] {time.time() - t0:.0f}s | loss {tr_loss:.4f} / {va_loss:.4f} | "
              + " ".join(f"{t} {v:.1f}" for t, v in zip(TARGETS, va_mae)))

        if va_mae[0] < best:  # model selection on validation calorie MAE
            best = float(va_mae[0])
            mean, std = scaler.to_dicts()
            torch.save({"model_state_dict": model.state_dict(), "target_cols": TARGETS,
                        "target_mean": mean, "target_std": std, "stage": args.stage,
                        "config": cfg, "epoch": epoch, "best_val_cal_mae": best}, out_path)
            print(f"   saved {out_path.name} (val calorie MAE {best:.2f} kcal)")

    log_dir = RESULTS_DIR / "training"
    log_dir.mkdir(parents=True, exist_ok=True)
    (log_dir / f"{args.stage}_history.json").write_text(json.dumps(history, indent=2))
    print(f"Done. Best val calorie MAE: {best:.2f} kcal. Evaluate with: python src/evaluate.py")


if __name__ == "__main__":
    main()
