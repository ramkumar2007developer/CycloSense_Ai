"""
train_numerical_only.py
=======================
Standalone script: trains the NumericalMLP on the environmental CSV dataset.

Scientific constraints:
- Group-aware storm-level train/val/test splitting (ADR-003). No row-level splits.
- StandardScaler fitted on train-only; val/test only transformed.
- Class-weighted CrossEntropyLoss for 87.7% / 12.3% imbalance (ADR-004).
- Metrics computed from actual predictions, not fabricated.
- single_class_warning raised if a split has only one class.
- data_source_tag: synthetic_mvp — results must NOT be presented as
  real meteorological observations.

This script does NOT touch the image CNN or fusion model.

Usage:
    python src/training/train_numerical_only.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from src.config.ml_config import load_ml_config
from src.data.numerical_pipeline import (
    build_numerical_pipeline,
    labels_from_frame,
    load_environmental_dataframe,
)
from src.data.splitting import split_by_group
from src.models.numerical_mlp import NumericalMLP
from src.training.metrics import compute_metrics
from src.training.trainer import train_numerical_from_arrays
from src.utils.reproducibility import set_random_seed


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _class_weights(y: np.ndarray) -> torch.Tensor:
    """Inverse-frequency class weights for imbalanced binary classification."""
    counts = np.bincount(y.astype(int), minlength=2).astype(float)
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
    print(f"    AUC-ROC  : {m.roc_auc:.4f}" if m.roc_auc is not None else "    AUC-ROC  : N/A (single class)")
    print(f"    Loss     : {m.loss:.4f}" if m.loss is not None else "")
    print(f"    Confusion: {m.confusion}  (rows=true, cols=pred, labels=[0,1])")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def train_numerical_only() -> None:
    print()
    print("=" * 70)
    print("  CYCLOSENSE AI — NUMERICAL-ONLY MLP TRAINING")
    print("  data_source_tag: synthetic_mvp")
    print("  ADR-003: group-aware split | ADR-004: class-weighted loss")
    print("  ADR-008: image classifier not involved")
    print("=" * 70)
    print()

    cfg = load_ml_config()
    set_random_seed(cfg.raw["training"]["seed"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # ── Data loading ─────────────────────────────────────────────────────────
    print("\nStep 1: Loading environmental CSV...")
    env_df = load_environmental_dataframe(cfg)
    print(f"  Loaded {len(env_df)} rows, {env_df['storm_id'].nunique()} unique storms.")

    # ── Group-aware split by storm_id ────────────────────────────────────────
    print("\nStep 2: Group-aware train/val/test split by storm_id (ADR-003)...")
    num_pipe = build_numerical_pipeline(cfg)
    split_cfg = cfg.raw["splitting"]
    splits = split_by_group(
        env_df,
        group_column=num_pipe.group_column,
        train_ratio=split_cfg["train_ratio"],
        val_ratio=split_cfg["val_ratio"],
        test_ratio=split_cfg["test_ratio"],
        seed=split_cfg["seed"],
    )
    print(f"  Train: {len(splits.train)} rows ({splits.train['storm_id'].nunique()} storms)")
    print(f"  Val:   {len(splits.validation)} rows ({splits.validation['storm_id'].nunique()} storms)")
    print(f"  Test:  {len(splits.test)} rows ({splits.test['storm_id'].nunique()} storms)")

    # Verify zero storm-level leakage
    train_storms = set(splits.train["storm_id"])
    val_storms = set(splits.validation["storm_id"])
    test_storms = set(splits.test["storm_id"])
    overlap_tv = train_storms & val_storms
    overlap_tt = train_storms & test_storms
    overlap_vt = val_storms & test_storms
    if overlap_tv or overlap_tt or overlap_vt:
        print(f"  ERROR: Storm leakage detected! TV={overlap_tv} TT={overlap_tt} VT={overlap_vt}",
              file=sys.stderr)
        sys.exit(1)
    print("  Leakage check: PASS (0 storm_id overlap across splits)")

    # ── Fit scaler on TRAIN only ─────────────────────────────────────────────
    print("\nStep 3: Fitting StandardScaler on train partition only (ADR-003)...")
    x_train = num_pipe.fit_transform(splits.train)
    x_val = num_pipe.transform(splits.validation)
    x_test = num_pipe.transform(splits.test)
    y_train = labels_from_frame(splits.train, num_pipe.label_column)
    y_val = labels_from_frame(splits.validation, num_pipe.label_column)
    y_test = labels_from_frame(splits.test, num_pipe.label_column)
    print(f"  Feature matrix shape: train={x_train.shape}, val={x_val.shape}, test={x_test.shape}")
    print(f"  Train label dist: {dict(zip(*np.unique(y_train, return_counts=True)))}")
    print(f"  Val   label dist: {dict(zip(*np.unique(y_val, return_counts=True)))}")
    print(f"  Test  label dist: {dict(zip(*np.unique(y_test, return_counts=True)))}")

    # ── Class weights ────────────────────────────────────────────────────────
    weights = _class_weights(y_train)
    print(f"\n  Class weights (inverse-freq): non-cyclone={weights[0]:.3f}, cyclone={weights[1]:.3f}")

    # ── Train NumericalMLP ────────────────────────────────────────────────────
    print("\nStep 4: Training NumericalMLP (2-layer, hidden_dim=64, dropout=0.1)...")
    train_cfg = cfg.raw["training"]
    model = NumericalMLP(input_dim=x_train.shape[1])
    result = train_numerical_from_arrays(
        model,
        x_train, y_train,
        x_val, y_val,
        epochs=train_cfg["epochs"],
        learning_rate=train_cfg["learning_rate"],
        batch_size=train_cfg["batch_size"],
        device=device,
        class_weights=weights,
    )
    print(f"  Training complete. Epochs={train_cfg['epochs']}")

    # ── Test evaluation ───────────────────────────────────────────────────────
    print("\nStep 5: Evaluating on held-out test partition...")
    model.eval()
    with torch.no_grad():
        x_test_t = torch.tensor(x_test, dtype=torch.float32, device=device)
        test_probs = torch.softmax(model(x_test_t), dim=1)[:, 1].cpu().numpy()
    test_metrics = compute_metrics(y_test, test_probs)

    # ── Print results ─────────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("  RESULTS — NumericalMLP (synthetic_mvp dataset)")
    print("=" * 70)
    _print_metrics("Train", result.train_metrics)
    print()
    _print_metrics("Val  ", result.val_metrics)
    print()
    _print_metrics("Test ", test_metrics)

    if result.val_metrics.single_class_warning:
        print(
            "\n  WARNING: Validation split has only one class.\n"
            "  Root cause: 87.7% cyclone imbalance + small val size.\n"
            "  Metrics are not reliable for this split. See ADR-004.",
            file=sys.stderr,
        )
    if test_metrics.single_class_warning:
        print(
            "\n  WARNING: Test split has only one class. Metrics not reliable.",
            file=sys.stderr,
        )

    # ── Save model and scaler ─────────────────────────────────────────────────
    print("\nStep 6: Saving model artifacts...")
    models_dir = PROJECT_ROOT / "models" / "numerical_model"
    models_dir.mkdir(parents=True, exist_ok=True)

    model_path = models_dir / "model.pt"
    torch.save(model.state_dict(), model_path)
    print(f"  Saved model weights: {model_path}")

    import joblib
    scaler_path = models_dir / "scaler.joblib"
    joblib.dump(num_pipe.scaler, scaler_path)
    print(f"  Saved scaler: {scaler_path}")

    # Save metrics JSON
    metrics_payload = {
        "data_source_tag": "synthetic_mvp",
        "model": "NumericalMLP",
        "architecture": "2-layer MLP, hidden_dim=64, dropout=0.1",
        "input_features": list(num_pipe.feature_columns),
        "epochs": train_cfg["epochs"],
        "learning_rate": train_cfg["learning_rate"],
        "batch_size": train_cfg["batch_size"],
        "class_weights": weights.tolist(),
        "split_sizes": {
            "train_rows": int(len(splits.train)),
            "val_rows": int(len(splits.validation)),
            "test_rows": int(len(splits.test)),
            "train_storms": int(splits.train["storm_id"].nunique()),
            "val_storms": int(splits.validation["storm_id"].nunique()),
            "test_storms": int(splits.test["storm_id"].nunique()),
        },
        "label_distribution": {
            "train": {int(k): int(v) for k, v in zip(*np.unique(y_train, return_counts=True))},
            "val": {int(k): int(v) for k, v in zip(*np.unique(y_val, return_counts=True))},
            "test": {int(k): int(v) for k, v in zip(*np.unique(y_test, return_counts=True))},
        },
        "val_metrics": result.val_metrics.as_dict(),
        "test_metrics": test_metrics.as_dict(),
        "WARNING": (
            "Results computed on synthetic_mvp dataset. "
            "Not validated meteorological predictions."
        ),
    }
    metrics_path = models_dir / "metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(metrics_payload, f, indent=2)
    print(f"  Saved metrics: {metrics_path}")

    print()
    print("=" * 70)
    print("  NUMERICAL-ONLY TRAINING COMPLETE")
    print(f"  Model:   {model_path}")
    print(f"  Scaler:  {scaler_path}")
    print(f"  Metrics: {metrics_path}")
    print("=" * 70)
    print()


if __name__ == "__main__":
    train_numerical_only()
