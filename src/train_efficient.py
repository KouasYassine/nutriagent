import torch
import torch.nn as nn
import pandas as pd
import  sklearn
from torchvision import models
from model import CalorieResNet
from dataset import NutritionDataset

df = pd.read_csv("data/processed/nutrition5k_labels.csv")
criterion = nn.SmoothL1Loss()

print(df.head())
print(df.columns)
from PIL import Image
import matplotlib.pyplot as plt


from torchvision import transforms

transform = transforms.Compose([
    transforms.Resize((320, 320)),
    transforms.RandomResizedCrop(299),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )])
row = df.iloc[0]

image = Image.open(row["image_path"]).convert("RGB")

image_tensor = transform(image)




from torch.utils.data import DataLoader



from sklearn.model_selection import train_test_split

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

calorie_mean = train_df["calories"].mean()
calorie_std = train_df["calories"].std()
dataset = NutritionDataset(df, transform=transform, target_mean=calorie_mean, target_std=calorie_std)
loader = DataLoader(
    dataset,
    batch_size=16,
    shuffle=True
)
print("Mean:", calorie_mean)
print("Std:", calorie_std)

images, calories = next(iter(loader))

print(images.shape)
print(calories.shape)
train_dataset = NutritionDataset(
    train_df,
    transform=transform,
    target_mean=calorie_mean,
    target_std=calorie_std
)

val_dataset = NutritionDataset(
    val_df,
    transform=transform,
    target_mean=calorie_mean,
    target_std=calorie_std
)

test_dataset = NutritionDataset(
    test_df,
    transform=transform,
    target_mean=calorie_mean,
    target_std=calorie_std
)
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
print(len(train_dataset))
print(len(val_dataset))
print(len(test_dataset))
images, calories = next(iter(train_loader))

print(images.shape)
print(calories.shape)


model = models.inception_v3(
    weights=models.Inception_V3_Weights.DEFAULT
)

# 1. Freeze everything
for param in model.parameters():
    param.requires_grad = False

# 2. Replace classification head
in_features = model.fc.in_features
model.fc = nn.Linear(in_features, 1)




device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

model = model.to(device)
# 3. create optimizer
optimizer = torch.optim.AdamW(
    model.fc.parameters(),
    lr=1e-4,
    weight_decay=1e-4)

total_loss = 0

for epoch in range(15):

    # ----- TRAIN -----
    model.train()
    train_loss = 0
    train_mae = 0
    for images, calories in train_loader:
        images = images.to(device)
        calories = calories.to(device)

        optimizer.zero_grad()

        predictions = model(images).squeeze(1)
        loss = criterion(predictions, calories)

        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        pred_kcal = predictions * calorie_std + calorie_mean
        true_kcal = calories * calorie_std + calorie_mean
        mae = torch.mean(torch.abs(pred_kcal - true_kcal))
        train_mae += mae.item()

    train_loss /= len(train_loader)
    train_mae /= len(train_loader)


    # ----- VALIDATION -----
    model.eval()
    val_loss = 0
    val_mae = 0

    with torch.no_grad():
        for images, calories in val_loader:
            images = images.to(device)
            calories = calories.to(device)

            predictions = model(images).squeeze(1)
            loss = criterion(predictions, calories)

            val_loss += loss.item()
            vpred_kcal = predictions * calorie_std + calorie_mean
            true_kcal = calories * calorie_std + calorie_mean

            val_mae += torch.mean(torch.abs(vpred_kcal - true_kcal))

    val_loss /= len(val_loader)
    val_mae /= len(val_loader)


    print(
        f"Epoch {epoch+1} | "
        f"Train Loss: {train_loss:.2f} | "
        f"train MAE: {train_mae:.2f} | "
        f"Val Loss: {val_loss:.2f}"
        f"val MAE: {val_mae:.2f}"
    )