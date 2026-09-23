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
from src.inference.llm_explainer import LLMExplainer, build_explainer
from src.inference.llm_verifier import LLMVisionVerifier, build_vision_verifier
from src.inference.verification_gate import (
    GateSignals,
    VerificationGateResult,
    evaluate_verification_gate,
)
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

    def _prepare_inputs(
        self, image_path: Path, env_row: pd.Series
    ) -> tuple[torch.Tensor, torch.Tensor, dict[str, float]]:
        """Validate and prepare image and numerical tensors."""
        image = preprocess_image(image_path, self.transform).unsqueeze(0)
        # Convert series or dict to DataFrame so num_pipe can validate columns
        row_dict = env_row.to_dict() if hasattr(env_row, "to_dict") else dict(env_row)
        env_df = pd.DataFrame([row_dict])
        scaled_features = self.num_pipe.transform(env_df)
        features = torch.tensor(scaled_features, dtype=torch.float32)
        env_dict = {col: float(row_dict[col]) for col in self.num_pipe.feature_columns}
        return image, features, env_dict

    @torch.no_grad()
    def extract_signals(self, image_path: Path, env_row: pd.Series) -> GateSignals:
        """Extract all 3 per-input prediction probabilities."""
        image, features, _ = self._prepare_inputs(image_path, env_row)
        img_logits = self.image_model(image)
        img_prob = float(torch.softmax(img_logits, dim=1)[0, 1].item())

        num_logits = self.num_model(features)
        num_prob = float(torch.softmax(num_logits, dim=1)[0, 1].item())

        fusion_logits = self.fusion_model(image, features)
        fusion_prob = float(torch.softmax(fusion_logits, dim=1)[0, 1].item())

        return GateSignals(
            image_probability=img_prob,
            numerical_probability=num_prob,
            fusion_probability=fusion_prob,
        )

    @torch.no_grad()
    def predict_image(self, image_path: Path) -> float:
        """Run isolated image inference."""
        image = preprocess_image(image_path, self.transform).unsqueeze(0)
        img_logits = self.image_model(image)
        return float(torch.softmax(img_logits, dim=1)[0, 1].item())

    @torch.no_grad()
    def predict_numerical(self, env_row: pd.Series) -> float:
        """Run isolated numerical inference."""
        row_dict = env_row.to_dict() if hasattr(env_row, "to_dict") else dict(env_row)
        env_df = pd.DataFrame([row_dict])
        scaled_features = self.num_pipe.transform(env_df)
        features = torch.tensor(scaled_features, dtype=torch.float32)
        num_logits = self.num_model(features)
        return float(torch.softmax(num_logits, dim=1)[0, 1].item())

    @torch.no_grad()
    def predict_fusion(self, image_path: Path, env_row: pd.Series) -> float:
        """Run isolated late-fusion inference."""
        image, features, _ = self._prepare_inputs(image_path, env_row)
        fusion_logits = self.fusion_model(image, features)
        return float(torch.softmax(fusion_logits, dim=1)[0, 1].item())

    @torch.no_grad()
    def predict(
        self,
        image_path: Path,
        env_row: pd.Series,
        verify: bool = False,
        verifier: LLMVisionVerifier | None = None,
        explain: bool = False,
        explainer: LLMExplainer | None = None,
    ) -> dict[str, object]:
        """Predict cyclone risk. Supports optional verification gate and LLM explanation."""
        if explain:
            return self.predict_with_explanation(
                image_path, env_row, verifier=verifier, explainer=explainer
            )
        if verify or self.config.llm_verification.enabled:
            return self.predict_with_verification(image_path, env_row, verifier=verifier)

        image, features, env_dict = self._prepare_inputs(image_path, env_row)
        fusion_logits = self.fusion_model(image, features)
        probs = torch.softmax(fusion_logits, dim=1)[0].cpu().numpy()
        risk = compute_risk_index(probs, env_dict, self.weights, data_source_tag=self.config.data_source_tag)
        return risk.as_dict()

    @torch.no_grad()
    def predict_with_verification(
        self,
        image_path: Path,
        env_row: pd.Series,
        verifier: LLMVisionVerifier | None = None,
        image_id: str | None = None,
    ) -> dict[str, object]:
        """Run ML prediction pipeline followed by the secondary LLM verification gate."""
        image, features, env_dict = self._prepare_inputs(image_path, env_row)

        # Extract all per-input probabilities
        img_logits = self.image_model(image)
        img_prob = float(torch.softmax(img_logits, dim=1)[0, 1].item())

        num_logits = self.num_model(features)
        num_prob = float(torch.softmax(num_logits, dim=1)[0, 1].item())

        fusion_logits = self.fusion_model(image, features)
        fusion_probs = torch.softmax(fusion_logits, dim=1)[0].cpu().numpy()
        fusion_prob = float(fusion_probs[1])

        signals = GateSignals(
            image_probability=img_prob,
            numerical_probability=num_prob,
            fusion_probability=fusion_prob,
        )

        risk = compute_risk_index(fusion_probs, env_dict, self.weights, data_source_tag=self.config.data_source_tag)
        risk_dict = risk.as_dict()

        # Build verifier if not passed
        actual_verifier = verifier
        if actual_verifier is None:
            actual_verifier = build_vision_verifier(self.config.llm_verification)

        gate_res: VerificationGateResult = evaluate_verification_gate(
            signals=signals,
            image_path=image_path,
            verifier=actual_verifier,
            config=self.config.llm_verification,
            environmental_context=env_dict,
            image_id=image_id or getattr(image_path, "name", "unknown"),
        )

        output: dict[str, object] = dict(risk_dict)
        output.update(gate_res.as_dict())
        return output

    @torch.no_grad()
    def predict_with_explanation(
        self,
        image_path: Path,
        env_row: pd.Series,
        verifier: LLMVisionVerifier | None = None,
        explainer: LLMExplainer | None = None,
        image_id: str | None = None,
    ) -> dict[str, object]:
        """Run ML prediction, verification gate, and generate structured meteorological explanation."""
        output = self.predict_with_verification(image_path, env_row, verifier=verifier, image_id=image_id)
        row_dict = env_row.to_dict() if hasattr(env_row, "to_dict") else dict(env_row)
        env_dict = {col: float(row_dict[col]) for col in self.num_pipe.feature_columns if col in row_dict}

        actual_explainer = explainer or build_explainer(self.config.llm_explanation)
        explanation = actual_explainer.generate_explanation(output, env_dict)
        output["llm_explanation"] = explanation.as_dict()
        return output
