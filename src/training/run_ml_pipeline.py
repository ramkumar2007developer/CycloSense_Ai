"""End-to-end ML/DL pipeline runner (no API)."""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from src.config.ml_config import MLConfig, load_ml_config
from src.data.audit import audit_datasets
from src.data.image_dataset import ImageOnlyDataset
from src.data.image_pipeline import build_image_transforms
from src.data.datasets import PairedCycloneDataset
from src.data.manifest import build_synthetic_mvp_manifest, save_manifest
from src.data.numerical_pipeline import (
    build_numerical_pipeline,
    labels_from_frame,
    load_environmental_dataframe,
)
from src.data.splitting import split_by_group
from src.models.fusion import FusionModel
from src.models.image_cnn import ImageBaselineCNN
from src.models.numerical_mlp import NumericalMLP
from src.training.metrics import MetricReport, compute_metrics
from src.training.trainer import train_model, train_numerical_from_arrays
from src.utils.reproducibility import set_random_seed


@dataclass
class PipelineArtifacts:
    audit: dict
    manifest_path: Path
    image_model_path: Path
    numerical_model_path: Path
    fusion_model_path: Path
    numerical_scaler_path: Path
    metrics_path: Path
    history_path: Path
    image_val_metrics: MetricReport
    numerical_val_metrics: MetricReport
    fusion_val_metrics: MetricReport
    image_test_metrics: MetricReport
    numerical_test_metrics: MetricReport
    fusion_test_metrics: MetricReport


def _class_weights(labels: np.ndarray) -> torch.Tensor:
    """Compute per-class weights to handle class imbalance (synthetic_mvp: 87.7% cyclone)."""
    counts = np.bincount(labels.astype(int), minlength=2).astype(float)
    counts[counts == 0] = 1.0
    weights = counts.sum() / (2.0 * counts)
    return torch.tensor(weights, dtype=torch.float32)


def _warn_single_class(split_name: str, metrics: MetricReport, model_name: str) -> None:
    """Print a clear warning when a split has only one class (known synthetic_mvp limitation)."""
    if metrics.single_class_warning:
        print(
            f"  ⚠ WARNING [{model_name} — {split_name}]: only one class in split. "
            "Accuracy/precision/F1 are unreliable. "
            "Root cause: synthetic_mvp 87.7% class-1 imbalance. "
            "This is a documented dataset limitation.",
            file=sys.stderr,
        )


def run_ml_pipeline(
    config: MLConfig | None = None,
    output_dir: Path | None = None,
) -> PipelineArtifacts:
    """Run the full ML/DL pipeline end-to-end.

    Steps:
    1. Audit datasets
    2. Build synthetic_mvp_pairing manifest
    3. Split by storm_id (no leakage)
    4. Fit numerical scaler on train only
    5. Train numerical MLP baseline
    6. Train image CNN baseline (MobileNetV3-Small)
    7. Train multimodal fusion model
    8. Evaluate all models on test set
    9. Save model weights, scaler, metrics, and training history

    NOTE: data_source_tag = 'synthetic_mvp'. Results must NOT be presented
    as scientifically validated meteorological predictions.
    """
    cfg = config or load_ml_config()
    set_random_seed(cfg.raw["training"]["seed"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    out = output_dir or cfg.project_root / "artifacts"
    models_dir = cfg.project_root / "models"
    out.mkdir(parents=True, exist_ok=True)
    models_dir.mkdir(parents=True, exist_ok=True)

    # --- Step 1: Audit ---
    print("Step 1: Auditing datasets...")
    audit = audit_datasets(cfg).to_dict()

    # --- Step 2: Manifest ---
    print("Step 2: Building synthetic_mvp_pairing manifest...")
    manifest = build_synthetic_mvp_manifest(cfg)
    manifest_path = save_manifest(manifest, out / "manifests" / "synthetic_mvp_pairing.csv")

    # --- Step 3: Split ---
    print("Step 3: Splitting data by storm_id (no leakage)...")
    env_df = load_environmental_dataframe(cfg)
    num_pipe = build_numerical_pipeline(cfg)
    split_cfg = cfg.raw["splitting"]
    env_splits = split_by_group(
        env_df,
        group_column=num_pipe.group_column,
        train_ratio=split_cfg["train_ratio"],
        val_ratio=split_cfg["val_ratio"],
        test_ratio=split_cfg["test_ratio"],
        seed=split_cfg["seed"],
    )
    manifest_splits = split_by_group(
        manifest,
        group_column="storm_id",
        train_ratio=split_cfg["train_ratio"],
        val_ratio=split_cfg["val_ratio"],
        test_ratio=split_cfg["test_ratio"],
        seed=split_cfg["seed"],
    )
    print(
        f"  Numerical splits — train: {len(env_splits.train)}, "
        f"val: {len(env_splits.validation)}, test: {len(env_splits.test)}"
    )
    print(
        f"  Image splits — train: {len(manifest_splits.train)}, "
        f"val: {len(manifest_splits.validation)}, test: {len(manifest_splits.test)}"
    )

    # --- Step 4: Fit scaler on TRAIN only ---
    print("Step 4: Fitting scaler on training data only (leakage prevention)...")
    x_train = num_pipe.fit_transform(env_splits.train)
    x_val = num_pipe.transform(env_splits.validation)
    x_test = num_pipe.transform(env_splits.test)
    y_train = labels_from_frame(env_splits.train, num_pipe.label_column)
    y_val = labels_from_frame(env_splits.validation, num_pipe.label_column)
    y_test = labels_from_frame(env_splits.test, num_pipe.label_column)
    print(f"  Train label distribution: {dict(zip(*np.unique(y_train, return_counts=True)))}")
    print(f"  Val label distribution:   {dict(zip(*np.unique(y_val, return_counts=True)))}")
    print(f"  Test label distribution:  {dict(zip(*np.unique(y_test, return_counts=True)))}")

    img_cfg = cfg.raw["image"]
    train_tf = build_image_transforms(
        img_cfg["size"], img_cfg["normalize_mean"], img_cfg["normalize_std"],
        augment=img_cfg["augment_train"],
    )
    eval_tf = build_image_transforms(
        img_cfg["size"], img_cfg["normalize_mean"], img_cfg["normalize_std"],
        augment=False,
    )
    train_cfg = cfg.raw["training"]
    weights = _class_weights(y_train)

    # --- Step 5: Numerical baseline ---
    print("Step 5: Training numerical MLP baseline...")
    num_model = NumericalMLP(input_dim=x_train.shape[1])
    num_result = train_numerical_from_arrays(
        num_model, x_train, y_train, x_val, y_val,
        epochs=train_cfg["epochs"], learning_rate=train_cfg["learning_rate"],
        batch_size=train_cfg["batch_size"], device=device, class_weights=weights,
    )
    num_model.eval()
    with torch.no_grad():
        test_prob = torch.softmax(
            num_model(torch.tensor(x_test, dtype=torch.float32, device=device)), dim=1
        )[:, 1].cpu().numpy()
    num_test_metrics = compute_metrics(y_test, test_prob)
    _warn_single_class("val", num_result.val_metrics, "NumericalMLP")
    _warn_single_class("test", num_test_metrics, "NumericalMLP")

    # --- Step 6: Image baseline ---
    print("Step 6: Training image CNN baseline (MobileNetV3-Small)...")
    img_model = ImageBaselineCNN(num_classes=2, pretrained=True)
    img_train_loader = DataLoader(
        ImageOnlyDataset(manifest_splits.train, train_tf),
        batch_size=train_cfg["batch_size"], shuffle=True,
        num_workers=train_cfg["num_workers"],
    )
    img_val_loader = DataLoader(
        ImageOnlyDataset(manifest_splits.validation, eval_tf),
        batch_size=train_cfg["batch_size"], num_workers=train_cfg["num_workers"],
    )
    img_test_loader = DataLoader(
        ImageOnlyDataset(manifest_splits.test, eval_tf),
        batch_size=train_cfg["batch_size"], num_workers=train_cfg["num_workers"],
    )
    img_result = train_model(
        img_model, img_train_loader, img_val_loader,
        epochs=train_cfg["epochs"], learning_rate=train_cfg["learning_rate"],
        device=device, modality="image", class_weights=weights,
    )
    y_img_test, p_img_test, img_test_loss = _eval_loader(img_model, img_test_loader, device, "image")
    img_test_metrics = compute_metrics(y_img_test, p_img_test, loss=img_test_loss)
    _warn_single_class("val", img_result.val_metrics, "ImageCNN")
    _warn_single_class("test", img_test_metrics, "ImageCNN")

    # --- Step 7: Fusion ---
    print("Step 7: Training multimodal fusion model...")
    fusion_model = FusionModel(num_features=len(num_pipe.feature_columns))
    fusion_train = DataLoader(
        PairedCycloneDataset(
            manifest_splits.train, env_df,
            num_pipe.feature_columns, num_pipe.label_column, train_tf,
        ),
        batch_size=train_cfg["batch_size"], shuffle=True,
        num_workers=train_cfg["num_workers"],
    )
    fusion_val = DataLoader(
        PairedCycloneDataset(
            manifest_splits.validation, env_df,
            num_pipe.feature_columns, num_pipe.label_column, eval_tf,
        ),
        batch_size=train_cfg["batch_size"], num_workers=train_cfg["num_workers"],
    )
    fusion_test = DataLoader(
        PairedCycloneDataset(
            manifest_splits.test, env_df,
            num_pipe.feature_columns, num_pipe.label_column, eval_tf,
        ),
        batch_size=train_cfg["batch_size"], num_workers=train_cfg["num_workers"],
    )
    fusion_result = train_model(
        fusion_model, fusion_train, fusion_val,
        epochs=train_cfg["epochs"], learning_rate=train_cfg["learning_rate"],
        device=device, modality="fusion", class_weights=weights,
    )
    y_f_test, p_f_test, f_test_loss = _eval_loader(fusion_model, fusion_test, device, "fusion")
    fusion_test_metrics = compute_metrics(y_f_test, p_f_test, loss=f_test_loss)
    _warn_single_class("val", fusion_result.val_metrics, "FusionModel")
    _warn_single_class("test", fusion_test_metrics, "FusionModel")

    # --- Step 8: Save artifacts ---
    print("Step 8: Saving model artifacts...")
    import joblib

    image_path = models_dir / "image_model" / "model.pt"
    numerical_path = models_dir / "numerical_model" / "model.pt"
    fusion_path = models_dir / "fusion_model" / "model.pt"
    scaler_path = out / "preprocessing" / "numerical_scaler.joblib"
    for p in (image_path, numerical_path, fusion_path):
        p.parent.mkdir(parents=True, exist_ok=True)
    scaler_path.parent.mkdir(parents=True, exist_ok=True)

    torch.save(img_model.state_dict(), image_path)
    torch.save(num_model.state_dict(), numerical_path)
    torch.save(fusion_model.state_dict(), fusion_path)
    joblib.dump(num_pipe.scaler, scaler_path)

    # Training history (per-epoch train_loss/val_loss)
    history = {
        "image": img_result.history,
        "numerical": num_result.history,
        "fusion": fusion_result.history,
    }
    history_path = out / "metrics" / "training_history.json"
    history_path.parent.mkdir(parents=True, exist_ok=True)
    history_path.write_text(json.dumps(history, indent=2), encoding="utf-8")

    # Full metrics (val + test)
    metrics = {
        "data_source_tag": cfg.data_source_tag,
        "pairing_tag": "synthetic_mvp_pairing",
        "image": {
            "val": img_result.val_metrics.as_dict(),
            "test": img_test_metrics.as_dict(),
            "train_vs_val_gap": {
                "accuracy": round(img_result.train_metrics.accuracy - img_result.val_metrics.accuracy, 4),
                "f1": round(img_result.train_metrics.f1 - img_result.val_metrics.f1, 4),
            },
        },
        "numerical": {
            "val": num_result.val_metrics.as_dict(),
            "test": num_test_metrics.as_dict(),
            "train_vs_val_gap": {
                "accuracy": round(num_result.train_metrics.accuracy - num_result.val_metrics.accuracy, 4),
                "f1": round(num_result.train_metrics.f1 - num_result.val_metrics.f1, 4),
            },
        },
        "fusion": {
            "val": fusion_result.val_metrics.as_dict(),
            "test": fusion_test_metrics.as_dict(),
            "train_vs_val_gap": {
                "accuracy": round(fusion_result.train_metrics.accuracy - fusion_result.val_metrics.accuracy, 4),
                "f1": round(fusion_result.train_metrics.f1 - fusion_result.val_metrics.f1, 4),
            },
        },
    }
    metrics_path = out / "metrics" / "pipeline_metrics.json"
    metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    print("Pipeline complete. All artifacts saved.")
    return PipelineArtifacts(
        audit=audit,
        manifest_path=manifest_path,
        image_model_path=image_path,
        numerical_model_path=numerical_path,
        fusion_model_path=fusion_path,
        numerical_scaler_path=scaler_path,
        metrics_path=metrics_path,
        history_path=history_path,
        image_val_metrics=img_result.val_metrics,
        numerical_val_metrics=num_result.val_metrics,
        fusion_val_metrics=fusion_result.val_metrics,
        image_test_metrics=img_test_metrics,
        numerical_test_metrics=num_test_metrics,
        fusion_test_metrics=fusion_test_metrics,
    )


@torch.no_grad()
def _eval_loader(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    modality: str,
) -> tuple[np.ndarray, np.ndarray, float]:
    """Evaluate a model on a DataLoader. Returns (y_true, y_prob, avg_loss)."""
    model.eval()
    probs: list[np.ndarray] = []
    labels: list[np.ndarray] = []
    total_loss = 0.0
    criterion = torch.nn.CrossEntropyLoss()
    for batch in loader:
        y = batch["label"].to(device)
        if modality == "image":
            logits = model(batch["image"].to(device))
        else:
            logits = model(batch["image"].to(device), batch["features"].to(device))
        total_loss += float(criterion(logits, y).item()) * y.size(0)
        probs.append(torch.softmax(logits, dim=1)[:, 1].cpu().numpy())
        labels.append(y.cpu().numpy())
    y_true = np.concatenate(labels)
    y_prob = np.concatenate(probs)
    return y_true, y_prob, total_loss / max(len(loader.dataset), 1)


if __name__ == "__main__":
    cfg = load_ml_config()
    print("Executing CycloSense ML/DL Training Pipeline...")
    artifacts = run_ml_pipeline(cfg)
    print("Pipeline execution complete!")
    print(f"Manifest: {artifacts.manifest_path}")
    print(f"Image Model: {artifacts.image_model_path}")
    print(f"Numerical Model: {artifacts.numerical_model_path}")
    print(f"Fusion Model: {artifacts.fusion_model_path}")
    print(f"Scaler: {artifacts.numerical_scaler_path}")
    print(f"Metrics: {artifacts.metrics_path}")
    print(f"History: {artifacts.history_path}")
    print(
        f"Fusion Test Accuracy: {artifacts.fusion_test_metrics.accuracy:.4f}, "
        f"F1: {artifacts.fusion_test_metrics.f1:.4f}"
    )

