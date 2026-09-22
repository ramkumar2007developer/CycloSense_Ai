# CycloSense AI — Dataset Notes & Provenance

This document details the data sources, preprocessing rules, limitations, and pairing strategies used in the CycloSense AI MVP.

---

## 1. Primary Data Sources

### A. Satellite Infrared Imagery
- **Source Directory:** `archive/insat3d_ir_cyclone_ds/CYCLONE_DATASET_INFRARED`
- **Total Files:** 136 image files (`.jpg`).
- **Data Integrity:** 0 corrupted images (verified via PIL file verification).
- **Color Format:** 3-channel RGB (pseudo-colored thermal infrared).
- **Resolution:** Variable dimensions ranging from ~166×166 to ~360×360 pixels.
- **Associated Metadata:** Accompanying CSV contains `img_name` and an integer `label` column (66 unique numeric grouping IDs).
- **Observation:** Every image in this specific archive folder is an active cyclone or tropical disturbance. There are **zero confirmed non-cyclone open-ocean images** in this directory.
- **Classifier Status:** See Section 5 — the image binary classifier **cannot** be scientifically trained or validated until genuine non-cyclone images are supplied. The image model is currently operating as an **embedding feature-extractor only**.

### B. Environmental Numerical Dataset
- **Source File:** `cyclosense_mvp_environmental_dataset.csv`
- **Total Records:** 1,000 observations across 100 unique simulated storms (`STORM_00` to `STORM_99`).
- **Data Source Tag:** Explicitly tagged as `synthetic_mvp`.
- **Target Variable:** `cyclone_label` (binary: 0 = non-cyclone, 1 = cyclone).
- **Class Distribution:** Severe class imbalance: **87.7% class 1 (cyclone)** and **12.3% class 0 (non-cyclone)**.
- **Feature Schema (12 numerical features):**
  1. `temperature_c` (expected: [15.0, 35.0] °C)
  2. `sea_surface_temperature_c` (expected: [20.0, 35.0] °C)
  3. `relative_humidity_pct` (expected: [0.0, 100.0] %)
  4. `water_vapour_gkg` (expected: [5.0, 30.0] g/kg)
  5. `surface_pressure_hpa` (expected: [800.0, 1100.0] hPa)
  6. `wind_speed_ms` (expected: [0.0, 100.0] m/s)
  7. `wind_direction_deg` (expected: [0.0, 360.0] °)
  8. `vertical_wind_shear_ms` (expected: [0.0, 50.0] m/s)
  9. `cloud_organization_index` (expected: [0.0, 1.0])
  10. `convection_index` (expected: [0.0, 1.0])
  11. `rotation_index` (expected: [0.0, 1.0])
  12. `persistence_index` (expected: [0.0, 1.0])

---

## 2. Pairing Strategy & Semantic Limitations

### The Pairing Problem
The synthetic environmental dataset contains synthetic image identifiers (e.g. `IMG_0001.jpg`), whereas the actual INSAT-3D infrared directory contains filenames like `25.jpg`, `104.jpg`, etc. There is **no physical or meteorological key linking the two sets**.

### The MVP Solution
- To validate the end-to-end multimodal architecture without corrupting scientific reality:
  - We generate a deterministic pairing manifest (`build_synthetic_mvp_manifest`).
  - Every paired row is stamped with `pairing_tag: "synthetic_mvp_pairing"` and `data_source_tag: "synthetic_mvp"`.
  - The image index is paired deterministically to environmental rows.
- **Scientific Caveat:** This pairing serves strictly to validate code execution, tensor dimension matching, gradient propagation, and data pipeline correctness. It must **not** be presented as a real observational study.

---

## 3. Data Leakage Prevention Guarantees

1. **Group Splitting:** Partitioning is performed strictly on `storm_id`. Any single storm's temporal sequence is entirely contained within either train (70%), validation (15%), or test (15%). No storm overlaps across splits.
2. **Train-Only Scaler Fitting:** `StandardScaler` is fitted solely on `train_df`. Validation and test partitions are transformed using parameters frozen from training.
3. **No Target Leakage:** Derived scoring indices (`development_risk_index`) and future time-series values are strictly excluded from the model input feature matrix.

---

## 4. Path to Production / Real-World Ingestion

For operational deployment beyond MVP:
- **ERA5 Reanalysis / GFS:** Ingest true reanalysis atmospheric soundings (ECMWF CDS API).
- **INSAT-3D / Himawari-8 Full-Disk / Rapid-Scan:** Ingest raw HDF5/netCDF4 calibrated brightness temperatures (IR 10.8 µm band) directly.
- **IBTrACS Integration:** Use official WMO best-track bulletins for exact storm track coordinates, minimum central pressure, and maximum sustained winds.

---

## 5. Image Classifier Limitation — Positive-Only Dataset (CRITICAL)

**Date documented:** 2026-09-22
**Documented by:** CycloSense AI Agent (Branch A decision, user-approved)

### Finding

All three INSAT-3D image archives in the provided dataset contain **exclusively cyclone images**:

| Folder | Images | Class Content |
|:---|:---:|:---|
| `insat3d_ir_cyclone_ds/CYCLONE_DATASET_INFRARED/` | 136 | All cyclone (positive-only) |
| `insat3d_for_reference_ds/CYCLONE_DATASET/` | 143 | All cyclone (positive-only, reference) |
| `insat3d_raw_cyclone_ds/CYCLONE_DATASET_FINAL/` | 139 | All cyclone (positive-only, raw) |

The CSV label column (`insat_3d_ds - Sheet.csv`) contains **storm group IDs** (integers
25–128), NOT binary cyclone/non-cyclone labels. This column cannot be reinterpreted as
a binary classification target.

**Total non-cyclone infrared images available: 0.**

### Scientific Consequence

A binary supervised classifier requires examples of BOTH classes. Training a CNN with
only positive (cyclone) samples produces a model that has learned "what a cyclone looks
like" but has no concept of "what a non-cyclone looks like". This guarantees:

1. The model will assign high cyclone probability to benign cloud formations.
2. Accuracy, precision, recall, and F1 metrics computed from positive-only data are
   **artefacts, not real performance metrics**. They must not be reported as such.
3. Any classification threshold tuning on positive-only data is meaningless.

This was empirically confirmed with `tests/probe_non_cyclone_inputs.py`:
- Cirrus streaks: p(cyclone) = 0.81
- Scattered cumulus: p(cyclone) = 0.80
- Checkerboard noise: p(cyclone) = 0.92

### Current Model Status

**`models/image_model/model.pt` is downgraded from binary classifier to embedding
extractor.** Its penultimate feature layer (before the classification head) will be
used to supply a 576-dimensional visual embedding to the multimodal fusion model.
The classification head output (p_cyclone / p_non_cyclone) is not used as a standalone
binary prediction.

### Condition for Reactivating Image Classification

The image binary classifier can be re-enabled when:
1. Genuine non-cyclone INSAT-3D infrared images are provided (minimum recommended: ~100 images).
2. Storm-level group IDs are assigned to non-cyclone images for group-aware splitting.
3. The model is retrained with both classes and evaluated with a held-out balanced test set.
4. `tests/probe_non_cyclone_inputs.py` reports p(cyclone) < 0.30 for all non-cyclone inputs.

### What Was NOT Done

- ❌ Fabricated non-cyclone images or synthetic negative labels.
- ❌ Hard-coded thresholds to artificially suppress false positives.
- ❌ Reported binary classification metrics on a positive-only dataset.
- ❌ Disguised the limitation behind a "multimodal dampener".
