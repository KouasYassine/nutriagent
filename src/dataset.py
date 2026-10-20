import torch
from torch.utils.data import Dataset
from PIL import Image


class NutritionDataset(Dataset):
    """
    Dataset for Experiment A1.

    Expected columns:
        image_path
        calories_scaled
        mass_scaled
        fat_scaled
        carbs_scaled
        protein_scaled

    Returns:
        image, target

    target order:
        [calories, mass, fat, carbs, protein]
    """

    def __init__(self, dataframe, transform=None):
        self.df = dataframe.reset_index(drop=True)
        self.transform = transform

        self.target_columns = [
            "calories_scaled",
            "mass_scaled",
            "fat_scaled",
            "carbs_scaled",
            "protein_scaled"
        ]

        missing_columns = [
            col for col in self.target_columns
            if col not in self.df.columns
        ]

        if missing_columns:
            raise ValueError(
                "Missing normalized target columns: "
                + ", ".join(missing_columns)
            )

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        image = Image.open(
            row["image_path"]
        ).convert("RGB")

        if self.transform is not None:
            image = self.transform(image)

        target = torch.tensor(
            [
                row["calories_scaled"],
                row["mass_scaled"],
                row["fat_scaled"],
                row["carbs_scaled"],
                row["protein_scaled"]
            ],
            dtype=torch.float32
        )

        return image, target
