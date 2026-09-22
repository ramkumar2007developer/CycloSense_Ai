"""Standalone prediction pipeline (no API)."""

from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch

from src.config.ml_config import MLConfig, load_ml_config
from src.data.image_pipeline import build_image_transforms, preprocess_image
from src.data.numerical_pipeline import build_numerical_pipeline
from src.models.fusion import FusionModel
from src.models.image_cnn import ImageBaselineCNN
from src.models.numerical_mlp import NumericalMLP
from src.scoring.risk_index import ScoringWeights, compute_risk_index


class CycloSensePredictor:
    def __init__(self, config: MLConfig, models_dir: Path, scaler_path: Path) -> None:
        self.config = config
        self.device = torch.device("cpu")
        self.num_pipe = build_numerical_pipeline(config)
        self.num_pipe.scaler = joblib.load(scaler_path)
        self.weights = ScoringWeights.from_config(config.raw["scoring"])
        img_cfg = config.raw["image"]
        self.transform = build_image_transforms(
            img_cfg["size"], img_cfg["normalize_mean"], img_cfg["normalize_std"], augment=False
        )
        n_features = len(self.num_pipe.feature_columns)
        self.image_model = ImageBaselineCNN(num_classes=2, pretrained=False)
        self.num_model = NumericalMLP(input_dim=n_features)
        self.fusion_model = FusionModel(num_features=n_features)
        self.image_model.load_state_dict(torch.load(models_dir / "image_model" / "model.pt", map_location="cpu"))
        self.num_model.load_state_dict(torch.load(models_dir / "numerical_model" / "model.pt", map_location="cpu"))
        self.fusion_model.load_state_dict(torch.load(models_dir / "fusion_model" / "model.pt", map_location="cpu"))
        for model in (self.image_model, self.num_model, self.fusion_model):
            model.eval()

    @classmethod
    def from_project(cls, project_root: Path | None = None) -> CycloSensePredictor:
        cfg = load_ml_config(project_root=project_root)
        root = cfg.project_root
        return cls(
            cfg,
            models_dir=root / "models",
            scaler_path=root / "artifacts" / "preprocessing" / "numerical_scaler.joblib",
        )

    @torch.no_grad()
    def predict(self, image_path: Path, env_row: pd.Series) -> dict[str, object]:
        image = preprocess_image(image_path, self.transform).unsqueeze(0)
        features_raw = env_row[self.num_pipe.feature_columns].astype(float).to_numpy(dtype=np.float32)
        features = torch.tensor(self.num_pipe.scaler.transform(features_raw.reshape(1, -1)), dtype=torch.float32)

        fusion_logits = self.fusion_model(image, features)
        probs = torch.softmax(fusion_logits, dim=1)[0].cpu().numpy()
        env_dict = {col: float(env_row[col]) for col in self.num_pipe.feature_columns}
        risk = compute_risk_index(probs, env_dict, self.weights, data_source_tag=self.config.data_source_tag)
        return risk.as_dict()
