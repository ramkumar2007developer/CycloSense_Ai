"""
train_fusion.py
===============
Standalone script: trains the multimodal FusionModel combining:
  - ImageBaselineCNN (embedding feature-extractor mode — NO classification head)
  - NumericalMLP (embedding mode)
  into a joint late-fusion classifier.

Scientific constraints and design decisions:
- ADR-007: ImageBaselineCNN is used in EMBEDDING EXTRACTOR mode only.
  The image model's classification head output (p_cyclone) is NOT used
  as a binary prediction. Only the 576-d penultimate pooled features are used.
  Rationale: the image dataset is positive-only (see DATASET_NOTES.md §5).
- ADR-008: The fusion model is trained on the binary cyclone_label from the
  environmental CSV (genuine 87.7%/12.3% labels), NOT on image-derived labels.
- ADR-003: Group-aware storm-level train/val/test split (no row-level random).
  Pairing manifest shares the same split seeds as the numerical training.
- ADR-004: Class-weighted CrossEntropyLoss for imbalance correction.
- data_source_tag: synthetic_mvp_pairing — image↔numerical pairing is
  deterministic but NOT physically grounded. Results must NOT be presented
  as real observational meteorological performance.

Usage:
    python src/training/train_fusion.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from src.config.ml_config import load_ml_config
from src.data.datasets import PairedCycloneDataset
from src.data.image_pipeline import build_image_transforms
from src.data.manifest import build_synthetic_mvp_manifest, save_manifest
from src.data.numerical_pipeline import (
    build_numerical_pipeline,
    load_environmental_dataframe,
)
from src.data.splitting import split_by_group
from src.models.fusion import FusionModel
from src.training.metrics import compute_metrics
from src.training.trainer import train_model
from src.utils.reproducibility import set_random_seed


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _class_weights(labels: np.ndarray) -> torch.Tensor:
    counts = np.bincount(labels.astype(int), minlength=2).astype(float)
    counts[counts == 0] = 1.0
    weights = counts.sum() / (2.0 * counts)
    return torch.tensor(weights, dtype=torch.float32)


def _print_metrics(label: str, m) -> None:
    warn = " [SINGLE-CLASS — METRICS UNRELIABLE]" if m.single_class_warning else ""
    print(f"  {label}{warn}")
    print(f"    Accuracy : {m.accuracy:.4f}")
    print(f"    Precision: {m.precision:.4f}")
    print(f"    Recall   : {m.recall:.4f}")
    print(f"    F1       : {m.f1:.4f}")
    if m.roc_auc is not None:
        print(f"    AUC-ROC  : {m.roc_auc:.4f}")
    else:
        print("    AUC-ROC  : N/A (single class)")
    if m.loss is not None:
        print(f"    Loss     : {m.loss:.4f}")
    print(f"    Confusion: {m.confusion}  (rows=true, cols=pred, labels=[0,1])")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def train_fusion() -> None:
    print()
    print("=" * 70)
    print("  CYCLOSENSE AI — FUSION MODEL TRAINING")
    print("  Image CNN (embedding-extractor) + NumericalMLP")
    print("  data_source_tag: synthetic_mvp_pairing")
    print("  ADR-007: image model in EMBEDDING mode (no classification head)")
    print("  ADR-008: trained on env CSV binary labels, not image labels")
    print("=" * 70)
    print()

    cfg = load_ml_config()
    set_random_seed(cfg.raw["training"]["seed"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # ── Build synthetic_mvp manifest ──────────────────────────────────────────
    print("\nStep 1: Building synthetic_mvp_pairing manifest...")
    manifest = build_synthetic_mvp_manifest(cfg)
    manifest_path = save_manifest(
        manifest,
        PROJECT_ROOT / "artifacts" / "manifests" / "fusion_synthetic_mvp_pairing.csv",
    )
    print(f"  Manifest: {len(manifest)} rows -> {manifest_path}")

    # ── Group-aware split (same seed as numerical training for consistency) ───
    print("\nStep 2: Group-aware train/val/test split by storm_id (ADR-003)...")
    split_cfg = cfg.raw["splitting"]
    manifest_splits = split_by_group(
        manifest,
        group_column="storm_id",
        train_ratio=split_cfg["train_ratio"],
        val_ratio=split_cfg["val_ratio"],
        test_ratio=split_cfg["test_ratio"],
        seed=split_cfg["seed"],
    )
    print(f"  Train: {len(manifest_splits.train)} rows "
          f"({manifest_splits.train['storm_id'].nunique()} storms)")
    print(f"  Val:   {len(manifest_splits.validation)} rows "
          f"({manifest_splits.validation['storm_id'].nunique()} storms)")
    print(f"  Test:  {len(manifest_splits.test)} rows "
          f"({manifest_splits.test['storm_id'].nunique()} storms)")

    # Leakage check
    tr = set(manifest_splits.train["storm_id"])
    va = set(manifest_splits.validation["storm_id"])
    te = set(manifest_splits.test["storm_id"])
    if (tr & va) or (tr & te) or (va & te):
        print("  ERROR: Storm leakage detected!", file=sys.stderr)
        sys.exit(1)
    print("  Leakage check: PASS (0 storm_id overlap across splits)")

    # ── Load environmental data for feature lookup ────────────────────────────
    print("\nStep 3: Loading environmental features (for feature column lookup)...")
    env_df = load_environmental_dataframe(cfg)
    num_pipe = build_numerical_pipeline(cfg)
    # Fit scaler on env train subset corresponding to manifest train storms
    env_train = env_df[env_df["storm_id"].isin(manifest_splits.train["storm_id"])]
    env_val = env_df[env_df["storm_id"].isin(manifest_splits.validation["storm_id"])]
    env_test = env_df[env_df["storm_id"].isin(manifest_splits.test["storm_id"])]

    # Scaler must be fit on train only
    x_train_arr = num_pipe.fit_transform(env_train)
    # We won't use these arrays directly (PairedCycloneDataset handles lookup),
    # but fitting the scaler here ensures it's frozen before any val/test access.
    # The scaler is stored in num_pipe.scaler, shared with PairedCycloneDataset.
    # NOTE: PairedCycloneDataset does NOT currently apply the scaler internally.
    # We pre-scale the env_df before passing to the dataset.
    feature_cols = num_pipe.feature_columns
    label_col = num_pipe.label_column

    # Scale the full env_df using the train-fitted scaler
    import pandas as pd
    scaled_train = pd.DataFrame(num_pipe.transform(env_train), columns=feature_cols,
                                index=env_train.index)
    scaled_train[label_col] = env_train[label_col].values
    scaled_train["storm_id"] = env_train["storm_id"].values
    scaled_train["timestamp"] = env_train["timestamp"].values

    scaled_val = pd.DataFrame(num_pipe.transform(env_val), columns=feature_cols,
                              index=env_val.index)
    scaled_val[label_col] = env_val[label_col].values
    scaled_val["storm_id"] = env_val["storm_id"].values
    scaled_val["timestamp"] = env_val["timestamp"].values

    scaled_test = pd.DataFrame(num_pipe.transform(env_test), columns=feature_cols,
                               index=env_test.index)
    scaled_test[label_col] = env_test[label_col].values
    scaled_test["storm_id"] = env_test["storm_id"].values
    scaled_test["timestamp"] = env_test["timestamp"].values

    y_train_arr = env_train[label_col].values
    print(f"  Train label dist: {dict(zip(*np.unique(y_train_arr, return_counts=True)))}")
    print(f"  Val   label dist: {dict(zip(*np.unique(env_val[label_col].values, return_counts=True)))}")
    print(f"  Test  label dist: {dict(zip(*np.unique(env_test[label_col].values, return_counts=True)))}")

    # ── Build image transforms ────────────────────────────────────────────────
    img_cfg = cfg.raw["image"]
    train_tf = build_image_transforms(
        img_cfg["size"], img_cfg["normalize_mean"], img_cfg["normalize_std"],
        augment=img_cfg["augment_train"],
    )
    eval_tf = build_image_transforms(
        img_cfg["size"], img_cfg["normalize_mean"], img_cfg["normalize_std"],
        augment=False,
    )

    # ── Build datasets and loaders ────────────────────────────────────────────
    print("\nStep 4: Building paired datasets...")
    train_cfg = cfg.raw["training"]

    train_ds = PairedCycloneDataset(
        manifest_splits.train, scaled_train,
        feature_columns=list(feature_cols), label_column=label_col,
        image_transform=train_tf,
    )
    val_ds = PairedCycloneDataset(
        manifest_splits.validation, scaled_val,
        feature_columns=list(feature_cols), label_column=label_col,
        image_transform=eval_tf,
    )
    test_ds = PairedCycloneDataset(
        manifest_splits.test, scaled_test,
        feature_columns=list(feature_cols), label_column=label_col,
        image_transform=eval_tf,
    )
    print(f"  Train dataset: {len(train_ds)} paired samples")
    print(f"  Val   dataset: {len(val_ds)} paired samples")
    print(f"  Test  dataset: {len(test_ds)} paired samples")

    train_loader = DataLoader(
        train_ds, batch_size=train_cfg["batch_size"],
        shuffle=True, num_workers=train_cfg["num_workers"],
    )
    val_loader = DataLoader(
        val_ds, batch_size=train_cfg["batch_size"],
        num_workers=train_cfg["num_workers"],
    )
    test_loader = DataLoader(
        test_ds, batch_size=train_cfg["batch_size"],
        num_workers=train_cfg["num_workers"],
    )

    # ── Build FusionModel ─────────────────────────────────────────────────────
    print("\nStep 5: Building FusionModel (image embedding + numerical embedding)...")
    print("  Image branch: ImageBaselineCNN.embedding() — 576-d pooled features ONLY")
    print("  Numerical branch: NumericalMLP.embedding() — 64-d hidden features")
    print("  Fusion head: Linear(256, 2)  [NOTE: ADR-007 — image head NOT used]")
    num_features = len(feature_cols)
    model = FusionModel(num_features=num_features)

    # Load pre-trained image backbone weights (embedding extractor only)
    image_model_path = PROJECT_ROOT / "models" / "image_model" / "model.pt"
    if image_model_path.exists():
        print(f"  Loading image backbone weights from {image_model_path}")
        state = torch.load(image_model_path, map_location="cpu")
        # Load into image encoder only (strict=False handles head mismatch)
        model.image_encoder.load_state_dict(state, strict=True)
        print("  Image backbone loaded. Classification head FROZEN (embedding-only).")
        # Freeze image classification head — only the embedding extractor is active
        # The fusion model's own classifier head will be trained from scratch.
        for param in model.image_encoder.model.classifier.parameters():
            param.requires_grad = False
        print("  Image classification head parameters frozen (grad=False).")
    else:
        print(f"  WARNING: image model not found at {image_model_path}. "
              "Using ImageNet pretrained backbone.", file=sys.stderr)

    # ── Class weights ─────────────────────────────────────────────────────────
    weights = _class_weights(y_train_arr)
    print(f"\n  Class weights: non-cyclone={weights[0]:.3f}, cyclone={weights[1]:.3f}")

    # ── Train ─────────────────────────────────────────────────────────────────
    print("\nStep 6: Training FusionModel...")
    result = train_model(
        model, train_loader, val_loader,
        epochs=train_cfg["epochs"],
        learning_rate=train_cfg["learning_rate"],
        device=device,
        modality="fusion",
        class_weights=weights,
    )
    print(f"  Training complete. Epochs={train_cfg['epochs']}")

    # ── Test evaluation ───────────────────────────────────────────────────────
    print("\nStep 7: Evaluating on held-out test partition...")
    from src.training.trainer import _predict
    y_test_arr, p_test_arr, test_loss = _predict(model, test_loader, device, "fusion")
    test_metrics = compute_metrics(y_test_arr, p_test_arr, loss=test_loss)

    # ── Print results ─────────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("  RESULTS — FusionModel (synthetic_mvp_pairing)")
    print("=" * 70)
    _print_metrics("Train", result.train_metrics)
    print()
    _print_metrics("Val  ", result.val_metrics)
    print()
    _print_metrics("Test ", test_metrics)

    if result.val_metrics.single_class_warning:
        print(
            "\n  WARNING: Validation split has only one class — metrics unreliable.",
            file=sys.stderr,
        )
    if test_metrics.single_class_warning:
        print(
            "\n  WARNING: Test split has only one class — metrics unreliable.",
            file=sys.stderr,
        )

    # ── Save artifacts ────────────────────────────────────────────────────────
    print("\nStep 8: Saving model artifacts...")
    models_dir = PROJECT_ROOT / "models" / "fusion_model"
    models_dir.mkdir(parents=True, exist_ok=True)

    model_path = models_dir / "model.pt"
    torch.save(model.state_dict(), model_path)
    print(f"  Saved model weights: {model_path}")

    import joblib
    scaler_path = models_dir / "scaler.joblib"
    joblib.dump(num_pipe.scaler, scaler_path)
    print(f"  Saved scaler: {scaler_path}")

    metrics_payload = {
        "data_source_tag": "synthetic_mvp_pairing",
        "model": "FusionModel",
        "image_branch": "ImageBaselineCNN (MobileNetV3-Small, embedding-extractor, 576-d)",
        "image_branch_status": (
            "EMBEDDING-EXTRACTOR ONLY (ADR-007). "
            "Classification head frozen. "
            "Positive-only image dataset — no binary image classification performed."
        ),
        "numerical_branch": "NumericalMLP (2-layer, hidden_dim=64, 64-d embedding)",
        "fusion": "concat([img_proj_128d, num_proj_128d]) -> Linear(256, 2)",
        "input_features": list(feature_cols),
        "epochs": train_cfg["epochs"],
        "learning_rate": train_cfg["learning_rate"],
        "batch_size": train_cfg["batch_size"],
        "class_weights": weights.tolist(),
        "split_sizes": {
            "train_rows": int(len(manifest_splits.train)),
            "val_rows": int(len(manifest_splits.validation)),
            "test_rows": int(len(manifest_splits.test)),
        },
        "label_distribution": {
            "train": {int(k): int(v) for k, v in zip(*np.unique(y_train_arr, return_counts=True))},
        },
        "val_metrics": result.val_metrics.as_dict(),
        "test_metrics": test_metrics.as_dict(),
        "WARNING": (
            "Results computed on synthetic_mvp_pairing dataset. "
            "Image↔numerical pairing is not physically grounded. "
            "Not validated meteorological predictions."
        ),
    }
    metrics_path = models_dir / "metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(metrics_payload, f, indent=2)
    print(f"  Saved metrics: {metrics_path}")

    print()
    print("=" * 70)
    print("  FUSION MODEL TRAINING COMPLETE")
    print(f"  Model:   {model_path}")
    print(f"  Scaler:  {scaler_path}")
    print(f"  Metrics: {metrics_path}")
    print("=" * 70)
    print()


if __name__ == "__main__":
    train_fusion()
