"""Image-only torch dataset."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import torch
from torch.utils.data import Dataset

from src.data.image_pipeline import preprocess_image


class ImageOnlyDataset(Dataset):
    def __init__(self, manifest: pd.DataFrame, transform) -> None:
        self.manifest = manifest.reset_index(drop=True)
        self.transform = transform

    def __len__(self) -> int:
        return len(self.manifest)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor | int | str]:
        row = self.manifest.iloc[index]
        image = preprocess_image(Path(row["image_path"]), self.transform)
        return {
            "image": image,
            "label": int(row["cyclone_label"]),
            "sample_id": str(row["sample_id"]),
        }
