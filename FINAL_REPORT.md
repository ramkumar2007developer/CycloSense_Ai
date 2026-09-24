# CycloSense AI: Comprehensive System Report

## 1. Executive Summary
CycloSense AI is an advanced, multimodal artificial intelligence platform designed for the detection, prediction, and explanation of cyclogenesis (tropical cyclone formation) and intensification. This report outlines the complete architecture of the system, which uniquely combines traditional Deep Learning/Machine Learning (DL/ML) for predictive accuracy with Large Language Models (LLMs) for secondary verification and human-readable meteorological explanations.

The platform processes both satellite imagery and numerical atmospheric data, generating quantitative risk indices and qualitative, actionable insights for meteorological forecasting.

---

## 2. System Architecture Overview

The CycloSense AI pipeline operates in four distinct stages:

1.  **Data Ingestion & Preprocessing:** Handles satellite image tensors and numerical atmospheric arrays (e.g., Sea Surface Temperature, Wind Shear).
2.  **Multimodal ML Inference:** Employs parallel DL/ML models to extract features and generate base predictions.
3.  **Secondary Verification Gate (LLM):** Acts as a safety layer to cross-reference ambiguous or high-risk ML predictions using an independent LLM Vision model.
4.  **Meteorological Explanation Generator (LLM):** Synthesizes all data (ML matrices, numerical inputs, LLM vision output) into structured, operational guidance.

---

## 3. Machine Learning (ML) Subsystem

The core predictive engine of CycloSense AI relies on a fused, multimodal approach.

### 3.1. Models
*   **Visual Signal Model (CNN):** Processes structural satellite imagery (e.g., INSAT/GOES) to identify cloud organization, spiral banding, and eye formation. It outputs a `Visual Signal` probability.
*   **Environmental Signal Model (MLP/Random Forest):** Analyzes numerical atmospheric parameters:
    *   Sea Surface Temperature (SST)
    *   Vertical Wind Shear
    *   Relative Humidity
    *   Atmospheric Pressure
    *   Vorticity and Convection Indices
    Outputs an `Environmental Signal` probability.
*   **Fusion Engine:** Concatenates embeddings from both the Visual and Environmental models through a final dense layer to output the **Multimodal Fusion Signal** and calculate the overall **Development/Risk Index (0-100)**.

### 3.2. ML Output Matrices
For a given input, the ML system produces:
*   `visual_prob`: Probability based purely on imagery.
*   `env_prob`: Probability based purely on atmospheric data.
*   `fusion_prob`: Combined probability.
*   `risk_index`: Scaled severity index.

---

## 4. LLM Secondary Verification Gate

To mitigate false positives and hallucinated risk scores, CycloSense AI implements a secondary verification gate (`LLMVisionVerifier`).

### 4.1. Trigger Condition
The verification gate is triggered **only** if all primary ML signals indicate a potential cyclone:
*   `visual_prob >= 0.50` AND `env_prob >= 0.50` AND `fusion_prob >= 0.50`

### 4.2. Verification Logic
If triggered, the raw satellite image is passed to a Vision-Language Model (e.g., Gemini Pro Vision) with a prompt asking it to independently assess cyclone structures (e.g., "Analyze this image for organized convection, banding, or an eye wall").
*   If the LLM confirms: Status becomes `VERIFIED_CYCLONE`.
*   If the LLM rejects: Status is downgraded to `UNVERIFIED_FALSE_POSITIVE`.

If the trigger condition is *not* met, the gate is bypassed, saving API costs and reducing latency.

---

## 5. LLM Explanation Generator

The final stage is the `LLMExplainer`, which bridges the gap between raw ML probabilities and human forecasters. It operates dynamically based on the verification gate's output.

### 5.1. Case A: High-Risk (Verified Cyclone)
When the ML system predicts a cyclone and the Verification Gate confirms it, the LLM generates a **High-Risk Explanation**.
*   **Inputs to LLM:** ML probabilities, Risk Index, specific numerical data (SST, wind shear), and the Verification Gate's vision findings.
*   **Output Structure:**
    *   *Summary:* High-level alert.
    *   *Atmospheric Rationale:* Why the ML model triggered.
    *   *Primary Drivers:* Factors fueling the storm (e.g., "SST is 30.5°C, providing immense latent heat").
    *   *Inhibiting Barriers:* Any factors working against the storm.
    *   *Operational Guidance:* Actionable advice (e.g., "Issue cyclone warning advisory").

### 5.2. Case B: Low-Risk / Suppressed
When the ML system predicts low probabilities, the Verification Gate is suppressed. The LLM generates a **Low-Risk Explanation**.
*   **Inputs to LLM:** ML probabilities and specific numerical data.
*   **Output Structure:**
    *   *Summary:* Routine update.
    *   *Atmospheric Rationale:* Why cyclogenesis is not expected.
    *   *Primary Drivers:* Irrelevant or weak factors.
    *   *Inhibiting Barriers:* The primary reasons for suppression (e.g., "High wind shear of 25m/s is tearing apart convective tops," or "Dry air intrusion").
    *   *Operational Guidance:* "Continue routine monitoring."

---

## 6. Testing and Validation

The system has undergone rigorous automated testing to ensure the stability of the multimodal pipeline and the deterministic behavior of the LLM integrations.

*   **Test Suite:** `pytest tests/`
*   **Coverage:** 98 tests covering ML inference, configuration, image processing, Verification Gate logic (all 11 edge cases), and Explanation Generation (all 9 edge cases).
*   **Mocking:** Full deterministic `Mock` providers are implemented for both the Verifier and Explainer to ensure tests pass without network dependencies or API keys.
*   **Result:** 98/98 tests passing successfully (0 regressions).

---

## 7. Execution and Deployment

### 7.1. Running the Pipeline
A dedicated CLI runner is available to evaluate the end-to-end pipeline:

```bash
python run_prediction.py
```

This script runs the full pipeline (ML → Verification → Explanation) on a high-risk test case (TC-01) and outputs the results, including the structured LLM explanation, to the terminal.

### 7.2. Configuration
API keys and model selection for the LLM subsystem are managed via `src/config/ml_config.py` and `configs/ml_config.json`.
Environment variables can override config files:
*   `CYCLOSENSE_LLM_PROVIDER`
*   `CYCLOSENSE_LLM_API_KEY`

---

## 8. Future Roadmap

1.  **Production LLM Integration:** Transition from Mock providers to live APIs (e.g., Google Gemini) by providing authorized production keys.
2.  **Dataset Expansion:** Add non-cyclone (negative) imagery to the training dataset to further harden the ML models against false positives.
3.  **Frontend/API Development:** Wrap the `CycloSensePredictor` in a RESTful API (e.g., FastAPI) and develop a web-based dashboard for forecasters to visualize the data and read the LLM explanations.
4.  **Temporal Sequences:** Upgrade the CNN to an LSTM or 3D-CNN to process sequences of satellite images over time, rather than single frames.
