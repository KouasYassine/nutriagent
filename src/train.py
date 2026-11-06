"""Train the multitask nutrition model.

Original experiments (reproduced exactly):
    python src/train.py --stage A1
    python src/train.py --stage A1_Phase2
    python src/train.py --stage A2_Mass2

New runs: give a --name and any combination of the improvements, e.g.
    python src/train.py --name v3_full --log-targets --img-size 384 --keep-aspect \
        --strong-aug --finetune full --head density --atwater-weight 0.1

Every run saves checkpoints/<name>.pth (with everything needed to rebuild the model)
and results/training/<name>_history.json. Model selection uses the validation
calorie MAE only; the test split is never touched here.
"""
import argparse
import json
import math
import random
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from config import CHECKPOINT_DIR, IMAGE_SIZE, RESULTS_DIR, SEED, TARGETS, load_labels, make_splits
from data import NutritionDataset, TargetScaler, build_transforms
from model import BACKBONES, MultiTaskResNet

# Original stages. A2_Mass2 matches the original A2 script exactly.
# A1 / A1_Phase2: check these against your original A1 code and edit if needed.
STAGES = {
    "A1": dict(init=None, finetune="heads", epochs=15, lr_backbone=0.0, lr=1e-3, mass_weight=1.0),
    "A1_Phase2": dict(init="A1", finetune="layer4", epochs=10, lr_backbone=1e-5, lr=1e-4, mass_weight=1.0),
    "A2_Mass2": dict(init="A1_Phase2", finetune="layer4", epochs=10, lr_backbone=5e-6, lr=5e-5, mass_weight=2.0),
}
I_CAL, I_FAT, I_CARB, I_PROT = (TARGETS.index(t) for t in ["calories", "fat", "carbs", "protein"])


def parse_args():
    p = argparse.ArgumentParser()
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("--stage", choices=STAGES, help="re-run one of the original experiments")
    g.add_argument("--name", help="name of a new run (checkpoint file name)")
    # model / data
    p.add_argument("--backbone", default="resnet18", choices=BACKBONES)
    p.add_argument("--head", default="direct", choices=["direct", "density"])
    p.add_argument("--img-size", type=int, default=IMAGE_SIZE)
    p.add_argument("--keep-aspect", action="store_true", help="keep the 4:3 aspect ratio (no squashing)")
    p.add_argument("--strong-aug", action="store_true", help="vertical flips + colour jitter")
    p.add_argument("--log-targets", action="store_true", help="regress log1p of every nutrient")
    # optimisation
    p.add_argument("--init", help="start from checkpoints/<init>.pth")
    p.add_argument("--finetune", default="full", choices=["heads", "layer4", "full"])
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--lr", type=float, default=1e-3, help="learning rate of the heads")
    p.add_argument("--lr-backbone", type=float, default=1e-4)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--warmup-epochs", type=float, default=1.0)
    p.add_argument("--mass-weight", type=float, default=1.0)
    p.add_argument("--atwater-weight", type=float, default=0.0,
                   help="penalty for calories != 4*protein + 4*carbs + 9*fat (0 = off)")
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--no-amp", action="store_true", help="disable mixed precision")
    p.add_argument("--seed", type=int, default=SEED)
    args = p.parse_args()

    if args.stage:  # original experiments: fixed recipe, old defaults
        s = STAGES[args.stage]
        args.name, args.init, args.finetune, args.epochs = args.stage, s["init"], s["finetune"], s["epochs"]
        args.lr, args.lr_backbone, args.mass_weight = s["lr"], s["lr_backbone"], s["mass_weight"]
        args.batch_size, args.warmup_epochs, args.no_amp = 16, 0.0, True
        args.backbone, args.head, args.img_size = "resnet18", "direct", IMAGE_SIZE
        args.keep_aspect = args.strong_aug = args.log_targets = False
        args.atwater_weight = 0.0
    if args.head == "density" and not args.log_targets:
        p.error("--head density requires --log-targets")
    return args


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def set_trainable(model, finetune):
    for prm in model.backbone.parameters():
        prm.requires_grad = finetune == "full"
    if finetune == "layer4":
        for prm in model.last_block().parameters():
            prm.requires_grad = True


def set_train_mode(model, finetune):
    model.train()
    if finetune != "full":
        # Frozen blocks keep their ImageNet BatchNorm statistics.
        model.backbone.eval()
        if finetune == "layer4":
            model.last_block().train()


def lr_lambda(total_steps, warmup_steps):
    """Linear warm-up then cosine decay to 0 (constant LR if there is no warm-up and 1 epoch)."""
    def f(step):
        if step < warmup_steps:
            return (step + 1) / warmup_steps
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return 0.5 * (1 + math.cos(math.pi * progress))
    return f


def compute_loss(outputs, targets, weights, scaler, atwater_weight):
    per_target = torch.stack([F.smooth_l1_loss(outputs[:, i], targets[:, i]) for i in range(len(TARGETS))])
    loss = (weights * per_target).sum()
    if atwater_weight > 0:
        real = scaler.inverse_torch(outputs.float()).clamp(min=0)
        atwater = 4 * real[:, I_PROT] + 4 * real[:, I_CARB] + 9 * real[:, I_FAT]
        # compared in log space so the penalty is scale-free (small and large plates count alike)
        loss = loss + atwater_weight * F.smooth_l1_loss(torch.log1p(real[:, I_CAL]), torch.log1p(atwater))
    return loss


def run_epoch(model, loader, weights, scaler, device, args, optimizer=None, sched=None, amp=None):
    training = optimizer is not None
    if training:
        set_train_mode(model, args.finetune)
    else:
        model.eval()
    total_loss, abs_err, n = 0.0, np.zeros(len(TARGETS)), 0
    use_amp = amp is not None and amp.is_enabled()
    with torch.set_grad_enabled(training):
        for images, targets in loader:
            images, targets = images.to(device, non_blocking=True), targets.to(device, non_blocking=True)
            with torch.autocast(device_type=device.type, enabled=use_amp):
                outputs = model(images)
            loss = compute_loss(outputs.float(), targets, weights, scaler, args.atwater_weight)
            if training:
                optimizer.zero_grad(set_to_none=True)
                amp.scale(loss).backward()
                amp.step(optimizer)
                amp.update()
                sched.step()
            total_loss += loss.item() * images.size(0)
            pred = np.clip(scaler.inverse(outputs.detach().float().cpu().numpy()), 0, None)
            true = scaler.inverse(targets.cpu().numpy())
            abs_err += np.abs(pred - true).sum(axis=0)
            n += images.size(0)
    return total_loss / n, abs_err / n


def main():
    args = parse_args()
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Run {args.name} | device {device}\n{json.dumps(vars(args), indent=1)}")

    train_df, val_df, _ = make_splits(load_labels())
    print(f"Train {len(train_df)} | Val {len(val_df)}")

    model = MultiTaskResNet(pretrained=args.init is None, backbone=args.backbone, head=args.head)
    if args.init:
        ckpt = torch.load(CHECKPOINT_DIR / f"{args.init}.pth", map_location="cpu", weights_only=True)
        model.load_state_dict(ckpt["model_state_dict"], strict=True)
        scaler = TargetScaler.from_checkpoint(ckpt)  # same normalisation across stages
    else:
        scaler = TargetScaler.fit(train_df, log=args.log_targets)
    if args.head == "density":
        model.set_target_stats(scaler.mean, scaler.std)
    model.to(device)

    set_trainable(model, args.finetune)
    groups = [{"params": model.head_parameters(), "lr": args.lr}]
    backbone_params = [p for p in model.backbone.parameters() if p.requires_grad]
    if backbone_params:
        groups.append({"params": backbone_params, "lr": args.lr_backbone})
    optimizer = torch.optim.AdamW(groups, weight_decay=args.weight_decay)

    kw = dict(batch_size=args.batch_size, num_workers=args.num_workers, pin_memory=device.type == "cuda",
              persistent_workers=args.num_workers > 0)
    tf = dict(img_size=args.img_size, keep_aspect=args.keep_aspect)
    train_loader = DataLoader(NutritionDataset(train_df, scaler, build_transforms(True, strong_aug=args.strong_aug, **tf)),
                              shuffle=True, drop_last=True, **kw)
    val_loader = DataLoader(NutritionDataset(val_df, scaler, build_transforms(False, **tf)), shuffle=False, **kw)

    steps = args.epochs * len(train_loader)
    if args.stage:  # original runs used a constant learning rate
        sched = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda s: 1.0)
    else:
        sched = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda(steps, int(args.warmup_epochs * len(train_loader))))
    amp = torch.amp.GradScaler("cuda", enabled=device.type == "cuda" and not args.no_amp)

    weights = torch.ones(len(TARGETS), device=device)
    weights[TARGETS.index("mass")] = args.mass_weight
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = CHECKPOINT_DIR / f"{args.name}.pth"
    model_config = dict(backbone=args.backbone, head=args.head, img_size=args.img_size,
                        keep_aspect=args.keep_aspect, strong_aug=args.strong_aug)
    best, history = float("inf"), []

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        tr_loss, tr_mae = run_epoch(model, train_loader, weights, scaler, device, args, optimizer, sched, amp)
        va_loss, va_mae = run_epoch(model, val_loader, weights, scaler, device, args)
        history.append({"epoch": epoch, "train_loss": tr_loss, "val_loss": va_loss,
                        **{f"train_mae_{t}": float(v) for t, v in zip(TARGETS, tr_mae)},
                        **{f"val_mae_{t}": float(v) for t, v in zip(TARGETS, va_mae)}})
        print(f"[{epoch:02d}/{args.epochs}] {time.time() - t0:.0f}s | loss {tr_loss:.4f} / {va_loss:.4f} | val MAE "
              + " ".join(f"{t} {v:.1f}" for t, v in zip(TARGETS, va_mae)))

        if va_mae[0] < best:
            best = float(va_mae[0])
            mean, std = scaler.to_dicts()
            torch.save({"model_state_dict": model.state_dict(), "target_cols": TARGETS,
                        "target_mean": mean, "target_std": std, "log_targets": scaler.log,
                        "model_config": model_config, "args": vars(args), "epoch": epoch,
                        "best_val_cal_mae": best}, out_path)
            print(f"   saved {out_path.name} (val calorie MAE {best:.2f} kcal)")

    log_dir = RESULTS_DIR / "training"
    log_dir.mkdir(parents=True, exist_ok=True)
    (log_dir / f"{args.name}_history.json").write_text(json.dumps(history, indent=2))
    print(f"Done. Best val calorie MAE: {best:.2f} kcal -> python src/evaluate.py --models {args.name} --tta")


if __name__ == "__main__":
    main()
