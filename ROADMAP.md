# CycloSense AI — Project Roadmap

CycloSense AI is an AI-assisted tropical cyclone monitoring research and MVP prototype combining satellite infrared imagery with numerical environmental fields.

---

## Phase Status Overview

| Phase | Description | Status | Gatekeeper Condition |
| :--- | :--- | :--- | :--- |
| **Phase 1** | Problem Formulation & Data Audit | **COMPLETE** | Full inspection of INSAT imagery & environmental CSV |
| **Phase 2** | Data Preprocessing & Validation Pipelines | **COMPLETE** | Out-of-range rejection, NaN checks, group splitting |
| **Phase 3** | Baseline & Multimodal Model Development | **COMPLETE** | Image CNN, Numerical MLP, Fusion model implemented |
| **Phase 4** | Rigorous Testing, Validation Gates & Anti-Bias | **IN PROGRESS / FINALIZING** | 74+ tests passing, leakage-free, edge cases verified |
| **Phase 5** | FastAPI Backend & Inference Engine | **BLOCKED** | Explicitly blocked until Phase 4 validation gates pass |
| **Phase 6** | Dashboard, Visualization & Explainability | **PENDING** | Depends on Phase 5 backend |
| **Phase 7** | Operational Pilot with Reanalysis Data | **FUTURE** | Requires real ERA5 & full IBTrACS integration |

---

## Phase Breakdown

### Phase 1: Problem Definition & Data Audit
- Conduct strict audit of all input data sources.
- Verify image integrity (corrupted JPGs, channel dimensions, resolution distributions).
- Audit numerical schema (columns, datatypes, missingness, physical boundary checks).
- Identify and document data ambiguities and pairing keys.

### Phase 2: Data Preprocessing & Leakage-Free Splitting
- Implement `group_split_by_storm` ensuring 0% storm overlap across train/val/test partitions.
- Implement strict train-only scaler fitting in `NumericalPipeline` to prevent test contamination.
- Implement deterministic image transformations and resizing (224×224).
- Add synthetic MVP pairing manifest generation with explicit data tags (`synthetic_mvp`, `synthetic_mvp_pairing`).

### Phase 3: Baseline & Multimodal Modeling
- **Image Baseline**: MobileNetV3-Small backbone for infrared cyclone visual classification.
- **Numerical Baseline**: 2-layer Multi-Layer Perceptron (MLP) with ReLU, Dropout, and feature encoder.
- **Multimodal Fusion**: Late-fusion architecture projecting visual embeddings (576-d) and numerical embeddings (64-d) into a unified classification head.
- **Scoring Pipeline**: Deterministic 0–100 CycloSense Development/Risk Index with component breakdowns.

### Phase 4: Rigorous Testing & Quality Gates (Current Phase)
- Implement comprehensive automated test suite (74+ tests).
- Validate zero data leakage (scalar invariance, partition disjointness).
- Verify model learning capacity via overfitting sanity checks.
- Enforce anti-cloud bias checks (normal cloud formations must not map to false high risk).
- Test all 10 edge cases and run invariant stress tests.
- Persist training history, confusion matrices, and ROC-AUC / F1 metrics.

### Phase 5: Backend API & Inference Service (Upcoming)
- Implement modular FastAPI application with Pydantic validation schemas.
- Expose `/predict/image`, `/predict/numerical`, `/predict/fusion`, and `/scoring/risk-index`.
- Ensure strict separation: inference engine imports ML modules without modifying training code.
- Return explicit `data_source_tag` and cautionary notes in all API payloads.

### Phase 6: Frontend & Explainability
- Develop interactive web UI for satellite image upload and environmental parameter sliders.
- Implement visual explanation layer (Grad-CAM heatmaps over infrared cyclone cloud structures).
- Implement feature attribution for numerical inputs (SHAP / Integrated Gradients).

### Phase 7: Real-World Atmospheric Pilot
- Ingest real ERA5 reanalysis soundings (ECMWF API).
- Process full-resolution INSAT-3D / Himawari-8 netCDF4 satellite streams.
- Validate predictions against official IMD / IBTrACS cyclone advisories.
