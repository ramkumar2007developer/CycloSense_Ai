"""PyTorch datasets for paired and numerical data."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import torch
from torch.utils.data import Dataset

from src.data.image_pipeline import preprocess_image


class PairedCycloneDataset(Dataset):
    def __init__(
        self,
        manifest: pd.DataFrame,
        env_df: pd.DataFrame,
        feature_columns: list[str],
        label_column: str,
        image_transform,
    ) -> None:
        self.manifest = manifest.reset_index(drop=True)
        self.env_df = env_df
        self.feature_columns = feature_columns
        self.label_column = label_column
        self.image_transform = image_transform

    def __len__(self) -> int:
        return len(self.manifest)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor | int | str]:
        row = self.manifest.iloc[index]
        image = preprocess_image(Path(row["image_path"]), self.image_transform)
        env_match = self.env_df[
            (self.env_df["storm_id"] == row["storm_id"])
            & (self.env_df["timestamp"] == row["timestamp"])
        ]
        if env_match.empty:
            env_row = self.env_df.iloc[index % len(self.env_df)]
        else:
            env_row = env_match.iloc[0]
        features = torch.tensor(
            env_row[self.feature_columns].astype(float).to_numpy(),
            dtype=torch.float32,
        )
        label = int(row["cyclone_label"])
        return {
            "image": image,
            "features": features,
            "label": label,
            "sample_id": str(row["sample_id"]),
        }


class NumericalDataset(Dataset):
    def __init__(self, features: torch.Tensor, labels: torch.Tensor) -> None:
        self.features = features
        self.labels = labels

    def __len__(self) -> int:
        return len(self.features)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        return {"features": self.features[index], "label": self.labels[index]}
