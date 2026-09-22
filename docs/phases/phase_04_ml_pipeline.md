# Phase 4 — Machine Learning & Deep Learning Pipeline Report

**Status:** Completed  
**Owner:** CycloSense AI Engineering Team  
**Phase Objective:** Design, build, train, test, and validate the end-to-end ML/DL pipeline for tropical cyclone detection and risk assessment before opening the FastAPI serving layer.

---

## 1. Architectural Architecture Overview

```
                                  +-----------------------------+
                                  |     INSAT-3D IR Images      |
                                  | (136 images, 224x224 RGB)   |
                                  +--------------+--------------+
                                                 |
                                                 v
                                  +-----------------------------+
                                  |  ImageBaselineCNN (MobileNet)|
                                  |   Encoder -> 576-d Latent   |
                                  +--------------+--------------+
                                                 |
                                                 v [Image Proj: 128-d]
+-------------------------------+                |
|  Synthetic Environmental CSV  |                |
|  (12 features, 1000 rows)     |                |
+---------------+---------------+                |
                |                                |
                v                                |
+-------------------------------+                |
| Scaler (fit on train only)    |                |
+---------------+---------------+                |
                |                                |
                v                                |
+-------------------------------+                |
|   NumericalMLP (2-layer MLP)  |                |
|   Encoder -> 64-d Latent      |                |
+---------------+---------------+                |
                |                                |
                v [Num Proj: 128-d]              |
                +----------------+---------------+
                                 |
                                 v [Concatenation: 256-d]
                +--------------------------------+
                |    Multimodal Fusion Head      |
                |    Linear(256, 2) -> Logits    |
                +----------------+---------------+
                                 |
                                 v
                +--------------------------------+
                |   Development / Risk Index     |
                |   (0-100 Composite Score)      |
                +--------------------------------+
```

---

## 2. Key Components Built

1. **`src/data/image_pipeline.py` & `src/data/image_dataset.py`:**
   - Standardizes satellite infrared images into $224 \times 224 \times 3$ PyTorch tensors.
   - Robust PIL verification catching corrupted or zero-byte files.
   - ImageNet normalizations applied deterministically.

2. **`src/data/numerical_pipeline.py`:**
   - Validates feature columns and rejects NaNs and out-of-range physical values.
   - Prevents data leakage via strict `fit(train_df)` before `transform()` enforcement.

3. **`src/data/splitting.py`:**
   - Implements `split_by_group(group_column="storm_id")`.
   - Strictly enforces zero storm overlap between train, validation, and test partitions.

4. **`src/models/`:**
   - `image_cnn.py`: MobileNetV3-Small backbone with embedding extraction.
   - `numerical_mlp.py`: 2-layer MLP with hidden dimension 64 and embedding extraction.
   - `fusion.py`: Late-fusion multimodal model projecting visual and numerical embeddings into a unified classifier.

5. **`src/scoring/risk_index.py`:**
   - Computes transparent 0–100 Development/Risk Index with component breakdowns (visual, environmental, convection, moisture, organization, persistence).
   - Labels output with `data_source_tag: "synthetic_mvp"` and explicit non-calibrated probability disclaimer.

6. **`src/training/`:**
   - `metrics.py`: Evaluates accuracy, precision, recall, F1, ROC-AUC, confusion matrices, and flags single-class splits (`single_class_warning`).
   - `trainer.py`: PyTorch training loops with Adam optimizer, class weighting, and loss logging.
   - `run_ml_pipeline.py`: Headless end-to-end execution pipeline saving models, scalers, manifests, and metrics.

---

## 3. Verification & Acceptance Criteria Review

All 17 acceptance criteria established in the engineering specification are satisfied:
1. Complete code audit conducted and documented.
2. Zero data leakage across splits (`test_leakage.py`).
3. Scaler parameters fitted solely on training partitions.
4. Input validation rejects NaNs, infinities, and out-of-range values.
5. All three models (Image CNN, Numerical MLP, Fusion) implemented and functional.
6. Models train without crashes and save state dictionaries.
7. Evaluation metrics computed from actual model predictions.
8. Confusion matrices recorded and inspected.
9. Single-class evaluation warnings explicitly handled.
10. Learning capacity verified on tiny fixed batches (> 50% loss reduction).
11. Anti-cloud bias tests verify non-cyclone formations do not trigger high risk.
12. All 10 domain edge cases verified with invariant stress tests.
13. Model perturbation robustness tests pass.
14. Save and load cycle produces identical predictions.
15. Full test suite (74 tests) passes automatically.
16. Synthetic MVP data tags clearly demarcated across manifests and scoring results.
17. FastAPI code completely deferred until this ML validation signoff.

---

## 4. Next Steps: Unlocking Phase 5 (FastAPI Serving)

With Phase 4 successfully completed and validated:
- The gate to **Phase 5 (FastAPI Backend)** is now cleared to proceed when requested.
- Saved model artifacts in `models/` and `artifacts/` can now be loaded into FastAPI prediction endpoints.
