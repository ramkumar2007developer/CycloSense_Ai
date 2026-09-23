# CycloSense AI — Architecture Decision Records (ADRs)

This document records the architectural and scientific design decisions made in CycloSense AI.

---

## ADR-001: Strict Separation Between ML Pipeline and API Backend
- **Status:** Accepted
- **Context:** Rushing to build an API before the ML pipeline is validated leads to serving uncalibrated, buggy, or leaked models.
- **Decision:** The project enforces an absolute phase boundary. No FastAPI or web serving code is written until the ML/DL pipeline passes all 17 acceptance criteria, achieves zero data leakage, and passes anti-bias checks.
- **Consequences:** Backend development is cleanly separated and consumes exported, versioned artifacts (`state_dict`, `scaler.joblib`, configs).

---

## ADR-002: Model Architectures
- **Status:** Accepted
- **Context:** Cyclone satellite imagery requires robust spatial pattern extraction, while environmental variables require nonlinear interaction modeling.
- **Decision:**
  - **Image Backbone:** MobileNetV3-Small (lightweight, efficient parameter count, fast inference, pre-trained on ImageNet for transfer learning).
  - **Numerical Architecture:** 2-layer MLP (hidden dimension 64, ReLU activation, 0.1 dropout) with dedicated embedding projection.
  - **Multimodal Fusion:** Late-fusion concatenation. The image embedding (576-d) and numerical embedding (64-d) are projected to equal dimensionality (128-d each) and passed to a joint linear classifier.
- **Consequences:** Low latency on CPU/GPU, modular upgrade path for replacing backbones (e.g., ConvNeXt or Swin Transformer) without changing the fusion interface.

---

## ADR-003: Group-Aware Splitting and Leakage Prevention
- **Status:** Accepted
- **Context:** Cyclone records from the same storm track over consecutive timesteps share strong autocorrelation. Random row-level splitting causes massive data leakage and artificially inflated evaluation metrics.
- **Decision:**
  - All dataset splits are partitioned by `storm_id` (`split_by_group`). 100% of observations from a given storm belong strictly to either train, validation, or test.
  - Feature normalization scalers (`StandardScaler`) are fitted exclusively on the training partition (`fit(train_df)`), never on validation or test sets. Calling `transform()` before `fit()` explicitly raises `DatasetValidationError`.
- **Consequences:** Evaluation metrics reflect genuine generalization to unseen storms.

---

## ADR-004: Explicit Handling and Tagging of Synthetic Data
- **Status:** Accepted
- **Context:** The MVP environmental CSV is synthetically generated (`synthetic_mvp`), and INSAT filenames do not naturally correlate with the synthetic storm IDs.
- **Decision:**
  - All artifacts, dataframes, manifests, and score payloads must carry an explicit `data_source_tag: "synthetic_mvp"` and `pairing_tag: "synthetic_mvp_pairing"`.
  - The code never disguises synthetic data as real meteorological observations.
  - When the validation split contains only a single class due to synthetic data distribution, a `single_class_warning` flag is raised instead of silently reporting misleading 100% metrics.
- **Consequences:** Transparent research and clinical/meteorological integrity.

---

## ADR-005: 0–100 Development/Risk Index Formulation
- **Status:** Accepted
- **Context:** End users need an interpretable, monotonic risk indicator, but raw neural network sigmoid outputs on small imbalanced datasets are not calibrated probabilities.
- **Decision:**
  - The CycloSense Development/Risk Index is formulated as a transparent weighted composite score (0 to 100) combining visual cyclone probability with key atmospheric parameters (organization, convection, rotation, persistence, moisture).
  - The score explicitly states in its metadata: `"Development/Risk Index — not a calibrated probability."`
- **Consequences:** Eliminates false certainty while providing actionable, explainable triage indicators.

---

## ADR-006: Anti-Cloud Bias Testing Protocol
- **Status:** Accepted
- **Context:** Deep learning models trained on satellite imagery frequently overfit to general cloud presence, erroneously predicting high cyclone risk for benign tropical cloud clusters or cirrus bands.
- **Decision:**
  - A dedicated anti-bias suite (`tests/ml/test_cloud_bias.py` and `tests/ml/test_edge_cases.py`) tests that clear sky, non-rotating cumulus, and unfavorable environmental shear do not trigger elevated risk indices.
- **Consequences:** Prevents false alarms in meteorological decision-support systems.

---

## ADR-007: Image Classifier Capability Downgrade — Positive-Only Dataset
- **Status:** Accepted
- **Date:** 2026-09-22
- **Context:** Empirical probe testing (`tests/probe_non_cyclone_inputs.py`) revealed that the trained `ImageBaselineCNN` assigns cyclone probability ≥ 0.80 to benign cloud formations (cirrus streaks, scattered cumulus, checkerboard noise). Root cause investigation confirmed that all 418 available INSAT-3D infrared images across all three archive folders are labelled "CYCLONE_DATASET". Zero confirmed non-cyclone images exist in the provided dataset. The CSV label column contains storm group IDs (integers), not binary cyclone/non-cyclone labels.
- **Decision:**
  - `models/image_model/model.pt` is **downgraded from binary classifier to embedding feature-extractor**.
  - The classification head output (`p_cyclone`, `p_non_cyclone`) is **not used as a standalone binary prediction** until genuine non-cyclone data is provided and the model is retrained.
  - The image model's penultimate 576-d feature layer is used solely to supply visual embeddings to the multimodal fusion model.
  - No fabricated negative samples, hard-coded thresholds, or suppressed probabilities are introduced.
  - The `run_image_sample_evaluation.py` evaluation script's pre-set low visual probabilities for non-cyclone cases are **retired** and replaced with documented embedding-mode outputs.
- **Consequences:** Scientific integrity is preserved. The system accurately represents what it can and cannot predict. Retraining with genuine negative data is unblocked whenever it becomes available.

---

## ADR-008: Numerical-Only and Fusion Model Training Proceed Independently
- **Status:** Accepted
- **Date:** 2026-09-22
- **Context:** The environmental numerical CSV (`cyclosense_mvp_environmental_dataset.csv`) has genuine binary labels (`cyclone_label`: 87.7% class 1, 12.3% class 0) and is independent of the image classification limitation.
- **Decision:**
  - Train the numerical-only 2-layer MLP independently using group-aware storm-level splitting and class-weighted loss.
  - Train a multimodal fusion model combining the image CNN's visual embeddings (feature-extractor mode) with the numerical MLP embedding. The fusion model is trained on the binary numerical labels, not on image-derived labels.
  - Both models are evaluated with precision, recall, F1, and AUC-ROC on the held-out test partition.
- **Consequences:** Two of the three planned models (numerical-only, fusion) can proceed without negative image data. The image binary classifier remains pending until negative data is available.

---

## ADR-009: Secondary LLM Vision Verification Gate and Decision Policy
- **Status:** Accepted
- **Date:** 2026-09-23
- **Context:** While the multimodal ML models output probabilities, empirical anti-bias probes revealed potential false positives on benign cloud formations (e.g. high cirrus bands, scattered cumulus, or non-rotating thunderstorms). A secondary visual verification layer is required before high-risk alerts are confirmed.
- **Decision:**
  - Introduce an automated, per-input secondary verification gate evaluated on every inference:
    `Gate Trigger: image_probability >= 0.50 AND numerical_probability >= 0.50 AND fusion_probability >= 0.50`
  - When all three per-input signals are $\ge 0.50$, the satellite image, model probabilities, and environmental context are presented to an LLM Vision Verifier (`LLMVisionVerifier`).
  - Strict anti-cloud bias prompting instructs the LLM: `"Do not assume that clouds indicate a cyclone. Normal scattered clouds, cirrus, dense cloud formations, convection, and other non-cyclonic atmospheric patterns can occur without a cyclone."`
  - The LLM does NOT silently override the ML model. The final status follows an explicit, documented policy:
    1. `VERIFIED_CYCLONE`: All signals $\ge 0.50$ AND LLM says `cyclone_consistent` AND confidence $\ge$ threshold (0.70).
    2. `SECONDARY_REVIEW_CONFLICT`: All signals $\ge 0.50$ BUT LLM says `non_cyclone_consistent` (flags false positives without forcing cyclone classification).
    3. `HUMAN_REVIEW_REQUIRED`: All signals $\ge 0.50$ AND (LLM says `uncertain` OR confidence $<$ threshold OR human review flagged).
    4. `LLM_NOT_REQUIRED`: Any signal $< 0.50$ (gate does not activate; normal ML result returned).
    5. `LLM_VERIFICATION_FAILED`: Gate activated but verifier encountered timeout, network error, or invalid response. System does not crash; original ML predictions are preserved.
- **Consequences:** Eliminates silent overrides, protects against cloud false positives, preserves full audit logging, and maintains complete testability with deterministic mocks when external credentials are not present.

---

## ADR-010: LLM Synthesis and Meteorological Explanation Generation
- **Status:** Accepted
- **Date:** 2026-09-23
- **Context:** Quantitative prediction matrices (probabilities, risk index 0–100, component weights, atmospheric indices) require actionable synthesis into natural language for operational triage and meteorological interpretation.
- **Decision:**
  - Introduce an `LLMExplainer` module to evaluate the complete matrix of predictions, environmental conditions, and verification results.
  - The explainer explicitly handles two distinct operational scenarios:
    1. **Case A (High Risk / Signals $\ge 50\%$ / Verification Triggered)**: Synthesizes visual spiral/eye features, numerical probabilities, and environmental parameters into a comprehensive verification rationale (`VERIFIED_CYCLONE`, `SECONDARY_REVIEW_CONFLICT`, or `HUMAN_REVIEW_REQUIRED`).
    2. **Case B (Other Case / Signals $< 50\%$ / Suppressed or Low Risk)**: Synthesizes atmospheric barriers (e.g. vertical wind shear, dry air, low SST, lack of rotation) explaining why development is suppressed and why verification was not required.
  - Return format is structured JSON containing: `summary`, `meteorological_rationale`, `primary_drivers`, `inhibiting_factors`, `operational_guidance`, and `case_type`.
  - On LLM failure or timeout, the explainer safely falls back to a deterministic, rule-based template without crashing the inference pipeline.
- **Consequences:** Provides interpretable, transparent, and auditable reasoning for both high-risk and fair-weather/suppressed cases without altering underlying model weights or probabilities.

