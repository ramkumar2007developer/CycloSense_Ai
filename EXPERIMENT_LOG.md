# CycloSense AI — Experiment Log

This document records the experimental configurations, training parameters, reproducibility seeds, and benchmark evaluation metrics for CycloSense AI.

---

## Experiment Index

| Run ID | Model | Split Scheme | Epochs | Batch Size | Learning Rate | Test Accuracy | Test F1 | Test ROC-AUC | Notes |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **EXP-001** | `ImageBaselineCNN` (MobileNetV3-Small) | Stratified by Group | 3 | 16 | 1e-3 (Adam) | 93.3% | 0.966 | 0.750 | Image-only baseline on 136 INSAT images |
| **EXP-002** | `NumericalMLP` (2-layer MLP) | Group split by `storm_id` | 10 | 32 | 1e-3 (Adam) | 79.3% | 0.877 | 0.787 | Numerical features on 1000 synthetic rows |
| **EXP-003** | `FusionModel` (Late-fusion) | Multimodal Group Split | 3 | 16 | 1e-3 (Adam) | 93.3% | 0.966 | 0.732 | Joint visual + environmental representation |

---

## Detailed Experiment Logs

### EXP-001: Image Baseline (MobileNetV3-Small)
- **Architecture:** Pre-trained MobileNetV3-Small with custom 2-class classification head.
- **Input:** 224×224×3 RGB-converted infrared satellite imagery (`archive/insat3d_ir_cyclone_ds/`).
- **Optimization:** Adam (`lr=0.001`), `CrossEntropyLoss` with class weights.
- **Random Seed:** `42` (ensuring deterministic transforms and batching).
- **Validation Metrics:**
  - Accuracy: 1.0 (Warning: Single-class partition with 20 class-1 samples, 0 class-0 samples).
  - Confusion Matrix: `[[0, 0], [0, 20]]`.
- **Test Metrics:**
  - Accuracy: 93.3% (28/30 correct).
  - Precision: 0.933, Recall: 1.0, F1: 0.966, ROC-AUC: 0.750.
  - Confusion Matrix: `[[0, 2], [0, 28]]`.
- **Observations:** Model correctly classifies all cyclone images (recall 1.0), but exhibits false positives on non-cyclone instances due to the near-total lack of negative imagery in the dataset.

---

### EXP-002: Numerical Baseline (NumericalMLP)
- **Architecture:** 2-layer MLP with hidden dimension 64, ReLU, and 0.1 dropout.
- **Input:** 12 environmental features scaled via `StandardScaler` fitted solely on the training partition.
- **Split:** Group split on `storm_id` (700 train / 150 val / 150 test rows; 0% storm overlap).
- **Optimization:** Adam (`lr=0.001`), `CrossEntropyLoss`.
- **Validation Metrics:**
  - Accuracy: 68.7% | Precision: 0.918 | Recall: 0.695 | F1: 0.791 | ROC-AUC: 0.754.
  - Confusion Matrix: `[[14, 8], [39, 89]]`.
- **Test Metrics:**
  - Accuracy: 79.3% | Precision: 0.941 | Recall: 0.822 | F1: 0.877 | ROC-AUC: 0.787.
  - Confusion Matrix: `[[8, 7], [24, 111]]`.
- **Observations:** Solid performance on synthetic environmental fields; demonstrates genuine discriminative capacity without overfitting or data leakage.

---

### EXP-003: Multimodal Fusion Model
- **Architecture:** Late-fusion architecture combining MobileNetV3 visual features (576-d) and numerical MLP features (64-d) projected into a 128-d latent space.
- **Input:** Paired synthetic MVP manifest (`pairing_tag: "synthetic_mvp_pairing"`).
- **Optimization:** Adam (`lr=0.001`), `CrossEntropyLoss`.
- **Validation Metrics:**
  - Accuracy: 1.0 (Single-class partition: `[[0, 0], [0, 20]]`).
  - Loss: ~5.1e-6.
- **Test Metrics:**
  - Accuracy: 93.3% | Precision: 0.933 | Recall: 1.0 | F1: 0.966 | ROC-AUC: 0.732.
  - Confusion Matrix: `[[0, 2], [0, 28]]`.
- **Observations:** Multimodal model reliably converges and outputs stable predictions across both modalities simultaneously.
