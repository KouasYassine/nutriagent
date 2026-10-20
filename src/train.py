import torch
import torch.nn as nn
import pandas as pd

from torchvision import transforms
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split

from model import MultiTaskResNet
from dataset import NutritionDataset


# ============================================================
# EXPERIMENT A2
# Mass-weighted multitask fine-tuning
#
# Starting point:
# best_multitask_resnet18_A1_phase2.pth
#
# Loss:
# L = L_cal + 2*L_mass + L_fat + L_carbs + L_protein
# ============================================================


# ============================================================
# 1. LOAD DATA
# ============================================================

df = pd.read_csv("data/processed/nutrition5k_labels.csv")

print(df.head())
print(df.columns)


# ============================================================
# 2. SAME TRAIN / VAL / TEST SPLIT
# ============================================================

train_df, temp_df = train_test_split(
    df,
    test_size=0.20,
    random_state=42
)

val_df, test_df = train_test_split(
    temp_df,
    test_size=0.50,
    random_state=42
)

print("Train:", len(train_df))
print("Validation:", len(val_df))
print("Test:", len(test_df))


# ============================================================
# 3. DEVICE
# ============================================================

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("Device:", device)


# ============================================================
# 4. LOAD BEST A1 PHASE 2 CHECKPOINT
# ============================================================

checkpoint_path = "best_multitask_resnet18_A1_phase2.pth"

checkpoint = torch.load(
    checkpoint_path,
    map_location=device
)

target_cols = checkpoint["target_cols"]

target_mean = pd.Series(
    checkpoint["target_mean"]
)[target_cols]

target_std = pd.Series(
    checkpoint["target_std"]
)[target_cols]

print("\nTarget means from checkpoint:")
print(target_mean)

print("\nTarget standard deviations from checkpoint:")
print(target_std)


# ============================================================
# 5. APPLY SAME NORMALIZATION
# ============================================================

train_df = train_df.copy()
val_df = val_df.copy()
test_df = test_df.copy()

for col in target_cols:

    train_df[f"{col}_scaled"] = (
        train_df[col] - target_mean[col]
    ) / target_std[col]

    val_df[f"{col}_scaled"] = (
        val_df[col] - target_mean[col]
    ) / target_std[col]

    test_df[f"{col}_scaled"] = (
        test_df[col] - target_mean[col]
    ) / target_std[col]


# ============================================================
# 6. IMAGE TRANSFORMS
# ============================================================

train_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(p=0.5),

    transforms.ToTensor(),

    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

eval_transform = transforms.Compose([
    transforms.Resize((224, 224)),

    transforms.ToTensor(),

    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])


# ============================================================
# 7. DATASETS
# ============================================================

train_dataset = NutritionDataset(
    train_df,
    transform=train_transform
)

val_dataset = NutritionDataset(
    val_df,
    transform=eval_transform
)

test_dataset = NutritionDataset(
    test_df,
    transform=eval_transform
)


# ============================================================
# 8. DATALOADERS
# ============================================================

train_loader = DataLoader(
    train_dataset,
    batch_size=16,
    shuffle=True
)

val_loader = DataLoader(
    val_dataset,
    batch_size=16,
    shuffle=False
)

test_loader = DataLoader(
    test_dataset,
    batch_size=16,
    shuffle=False
)

print("Train dataset:", len(train_dataset))
print("Validation dataset:", len(val_dataset))
print("Test dataset:", len(test_dataset))


# ============================================================
# 9. MODEL
# ============================================================

model = MultiTaskResNet()

model.load_state_dict(
    checkpoint["model_state_dict"]
)

model = model.to(device)


# ============================================================
# 10. FREEZE BACKBONE EXCEPT LAYER4
# ============================================================

for param in model.backbone.parameters():
    param.requires_grad = False

for param in model.backbone.layer4.parameters():
    param.requires_grad = True

print("\nTrainable parameters:")

for name, param in model.named_parameters():
    if param.requires_grad:
        print(name)


# ============================================================
# 11. LOSS
# ============================================================

criterion = nn.SmoothL1Loss()


# ============================================================
# 12. OPTIMIZER
#
# Lower LR than Phase 2 because we are now fine-tuning
# an already trained model.
# ============================================================

optimizer = torch.optim.AdamW([
    {
        "params": model.backbone.layer4.parameters(),
        "lr": 5e-6
    },
    {
        "params": model.shared.parameters(),
        "lr": 5e-5
    },
    {
        "params": model.calories_head.parameters(),
        "lr": 5e-5
    },
    {
        "params": model.mass_head.parameters(),
        "lr": 5e-5
    },
    {
        "params": model.fat_head.parameters(),
        "lr": 5e-5
    },
    {
        "params": model.carbs_head.parameters(),
        "lr": 5e-5
    },
    {
        "params": model.protein_head.parameters(),
        "lr": 5e-5
    }
], weight_decay=1e-4)


# ============================================================
# 13. HELPERS
# ============================================================

target_mean_t = torch.tensor(
    target_mean.values,
    dtype=torch.float32,
    device=device
)

target_std_t = torch.tensor(
    target_std.values,
    dtype=torch.float32,
    device=device
)


def denormalize_targets(x):
    return x * target_std_t + target_mean_t


def compute_mae_sums(outputs, targets):

    outputs_real = denormalize_targets(outputs)
    targets_real = denormalize_targets(targets)

    return torch.abs(
        outputs_real - targets_real
    ).sum(dim=0)


# ============================================================
# 14. TRAINING SETTINGS
# ============================================================

num_epochs = 10

best_val_cal_mae = float("inf")
best_val_mass_mae = float("inf")

output_checkpoint = (
    "best_multitask_resnet18_A2_mass2.pth"
)


# ============================================================
# 15. TRAINING
# ============================================================

for epoch in range(num_epochs):

    # ========================================================
    # TRAIN
    # ========================================================

    model.train()

    # Keep frozen backbone BatchNorm statistics fixed.
    model.backbone.eval()

    # Only layer4 should behave as trainable backbone block.
    model.backbone.layer4.train()

    train_loss = 0.0
    train_abs_error_sums = torch.zeros(
        5,
        device=device
    )
    train_samples = 0

    for images, targets in train_loader:

        images = images.to(device)
        targets = targets.to(device)

        optimizer.zero_grad()

        outputs = model(images)

        # ----------------------------------------------------
        # SEPARATE TASK LOSSES
        # ----------------------------------------------------

        loss_cal = criterion(
            outputs[:, 0],
            targets[:, 0]
        )

        loss_mass = criterion(
            outputs[:, 1],
            targets[:, 1]
        )

        loss_fat = criterion(
            outputs[:, 2],
            targets[:, 2]
        )

        loss_carbs = criterion(
            outputs[:, 3],
            targets[:, 3]
        )

        loss_protein = criterion(
            outputs[:, 4],
            targets[:, 4]
        )

        # ----------------------------------------------------
        # A2 MASS-WEIGHTED MULTITASK LOSS
        # ----------------------------------------------------

        loss = (
            loss_cal
            + 2.0 * loss_mass
            + loss_fat
            + loss_carbs
            + loss_protein
        )

        loss.backward()
        optimizer.step()

        train_loss += loss.item()

        train_abs_error_sums += compute_mae_sums(
            outputs.detach(),
            targets
        )

        train_samples += images.size(0)

    train_loss /= len(train_loader)

    train_maes = (
        train_abs_error_sums / train_samples
    ).detach().cpu()


    # ========================================================
    # VALIDATION
    # ========================================================

    model.eval()

    val_loss = 0.0

    val_abs_error_sums = torch.zeros(
        5,
        device=device
    )

    val_samples = 0

    with torch.no_grad():

        for images, targets in val_loader:

            images = images.to(device)
            targets = targets.to(device)

            outputs = model(images)

            loss_cal = criterion(
                outputs[:, 0],
                targets[:, 0]
            )

            loss_mass = criterion(
                outputs[:, 1],
                targets[:, 1]
            )

            loss_fat = criterion(
                outputs[:, 2],
                targets[:, 2]
            )

            loss_carbs = criterion(
                outputs[:, 3],
                targets[:, 3]
            )

            loss_protein = criterion(
                outputs[:, 4],
                targets[:, 4]
            )

            loss = (
                loss_cal
                + 2.0 * loss_mass
                + loss_fat
                + loss_carbs
                + loss_protein
            )

            val_loss += loss.item()

            val_abs_error_sums += compute_mae_sums(
                outputs,
                targets
            )

            val_samples += images.size(0)

    val_loss /= len(val_loader)

    val_maes = (
        val_abs_error_sums / val_samples
    ).detach().cpu()

    val_cal_mae = val_maes[0].item()
    val_mass_mae = val_maes[1].item()


    # ========================================================
    # SAVE BEST MODEL
    #
    # Primary criterion stays calories.
    # We also print mass so we can judge the tradeoff.
    # ========================================================

    if val_cal_mae < best_val_cal_mae:

        best_val_cal_mae = val_cal_mae
        best_val_mass_mae = val_mass_mae

        torch.save(
            {
                "model_state_dict": model.state_dict(),
                "target_mean": target_mean.to_dict(),
                "target_std": target_std.to_dict(),
                "target_cols": target_cols,
                "mass_loss_weight": 2.0,
                "best_val_cal_mae": best_val_cal_mae,
                "best_val_mass_mae": best_val_mass_mae
            },
            output_checkpoint
        )

        print(
            f"\nNew best A2 model saved! "
            f"Val calorie MAE: "
            f"{best_val_cal_mae:.2f} kcal | "
            f"Val mass MAE: "
            f"{best_val_mass_mae:.2f} g"
        )


    # ========================================================
    # DISPLAY
    # ========================================================

    print(
        f"\nEpoch {epoch + 1:02d}/{num_epochs} | "
        f"Train Loss: {train_loss:.4f} | "
        f"Val Loss: {val_loss:.4f}"
    )

    print(
        "Train MAE | "
        f"Calories: {train_maes[0]:.2f} kcal | "
        f"Mass: {train_maes[1]:.2f} g | "
        f"Fat: {train_maes[2]:.2f} g | "
        f"Carbs: {train_maes[3]:.2f} g | "
        f"Protein: {train_maes[4]:.2f} g"
    )

    print(
        "Val   MAE | "
        f"Calories: {val_maes[0]:.2f} kcal | "
        f"Mass: {val_maes[1]:.2f} g | "
        f"Fat: {val_maes[2]:.2f} g | "
        f"Carbs: {val_maes[3]:.2f} g | "
        f"Protein: {val_maes[4]:.2f} g"
    )


# ============================================================
# 16. FINAL MESSAGE
# ============================================================

print(
    "\nA2 complete."
)

print(
    f"Best validation calorie MAE: "
    f"{best_val_cal_mae:.2f} kcal"
)

print(
    f"Mass MAE at that checkpoint: "
    f"{best_val_mass_mae:.2f} g"
)

print(
    f"Checkpoint saved as: "
    f"{output_checkpoint}"
)

print(
    "\nReference from A1 Phase 2:"
)

print(
    "Calories MAE = 60.48 kcal"
)

print(
    "Mass MAE = 43.06 g"
)

print(
    "\nDo not evaluate on the test set yet."
)
