# CycloSense AI — FastAPI Backend Documentation

## 1. Overview & Architecture

The CycloSense AI FastAPI backend provides a high-performance RESTful API serving the multimodal cyclogenesis detection, risk index scoring, secondary LLM verification gate, and meteorological explanation generation pipeline.

### System Pipeline

```
[Satellite Infrared Image] + [12 Numerical Atmospheric Variables]
                       ↓
              FastAPI Inference Layer
            (POST /predict - multipart)
                       ↓
               CycloSensePredictor
      ┌─────────────────────────────┐
      │  Image Baseline CNN         │ ── (Visual Signal Probability)
      │  Numerical 2-layer MLP      │ ── (Environmental Signal Probability)
      │  Late-Fusion Model          │ ── (Fusion Signal Probability)
      └─────────────────────────────┘
                       ↓
       CycloSense Development/Risk Index (0–100)
                       ↓
       Secondary Verification Gate (>= 50% Threshold)
        ├── All signals >= 0.50 ──> Invoke LLM Vision Verification
        └── Any signal < 0.50  ──> Verification Suppressed (LLM Not Required)
                       ↓
       LLM Meteorological Explanation Generator
        ├── Case A (High Risk / Verified)
        └── Case B (Suppressed / Fair Weather / Hostile Shear / Low SST)
                       ↓
            Structured JSON Response
```

### Key Architectural Characteristics
- **Singleton Lifespan Loading**: PyTorch models (`ImageBaselineCNN`, `NumericalMLP`, `FusionModel`) and Scikit-Learn feature scalers (`StandardScaler`) are pre-warmed once upon server startup and cached in memory. Requests execute inference directly without reloading weights from disk.
- **Strict Boundary Validation**: Pydantic input models enforce physical meteorological limits matching `FEATURE_RANGES`.
- **Ephemeral Storage Hygiene**: Uploaded images are validated in-memory (magic bytes, dimensions, Pillow color mode), processed through temporary files, and unlinked inside guaranteed `finally` blocks.
- **Zero Secret Exposure**: Model configuration reads from system environment variables; secrets and API keys are never included in logs, exceptions, or JSON payloads.

---

## 2. API Endpoints

### 2.1. Health Check
- **Path**: `GET /health`
- **Description**: Returns the operational status and service version without invoking ML models.
- **Response**: `200 OK`
```json
{
  "status": "ok",
  "service": "CycloSense AI",
  "version": "0.1.0"
}
```

---

### 2.2. Multimodal Prediction & Explanation
- **Path**: `POST /predict`
- **Content-Type**: `multipart/form-data`
- **Description**: Evaluates an uploaded satellite infrared image alongside 12 atmospheric parameters, calculates the composite Development/Risk Index, runs secondary verification, and generates natural-language meteorological explanations.

#### Request Parameters (Multipart Form)

| Parameter | Type | Required | Range / Format | Description |
| :--- | :--- | :--- | :--- | :--- |
| `image` | File | Yes | `.jpg`, `.jpeg`, `.png` ($\le 10$ MB) | Satellite infrared image |
| `case_id` | String | No | Default: `"custom"` | Case identifier (e.g. `"TC-01"`) |
| `temperature_c` | Float | Yes | `[-100.0, 60.0]` | Ambient air temperature in °C |
| `sea_surface_temperature_c` | Float | Yes | `[-5.0, 45.0]` | Sea surface temperature in °C |
| `relative_humidity_pct` | Float | Yes | `[0.0, 100.0]` | Mid-tropospheric relative humidity (%) |
| `water_vapour_gkg` | Float | Yes | `[0.0, 100.0]` | Specific humidity / water vapour (g/kg) |
| `surface_pressure_hpa` | Float | Yes | `[800.0, 1100.0]` | Atmospheric surface pressure (hPa) |
| `wind_speed_ms` | Float | Yes | `[0.0, 120.0]` | Sustained surface wind speed (m/s) |
| `wind_direction_deg` | Float | Yes | `[0.0, 360.0]` | Wind direction azimuth (degrees) |
| `vertical_wind_shear_ms` | Float | Yes | `[0.0, 50.0]` | 850–200 hPa vertical wind shear (m/s) |
| `cloud_organization_index` | Float | Yes | `[0.0, 1.0]` | Normalized cloud canopy organization |
| `convection_index` | Float | Yes | `[0.0, 1.0]` | Normalized deep convective intensity |
| `rotation_index` | Float | Yes | `[0.0, 1.0]` | Normalized low-level vorticity / rotation |
| `persistence_index` | Float | Yes | `[0.0, 1.0]` | Normalized disturbance persistence |

---



### 2.3. Isolated Sub-Model Inferences

The API also exposes the individual ML models for decoupled usage by the frontend or external systems.

- **`POST /predict/image`**
  - **Description**: Runs only the MobileNetV3 Image CNN.
  - **Inputs**: `image` (File), `case_id` (String)
  - **Returns**: `ImagePredictionResponse` containing `visual_signal` (0.0 - 1.0).

- **`POST /predict/numerical`**
  - **Description**: Runs only the 2-layer Numerical MLP.
  - **Inputs**: 12 atmospheric parameters via form data, `case_id` (String)
  - **Returns**: `NumericalPredictionResponse` containing `environmental_signal` (0.0 - 1.0).

- **`POST /predict/fusion`**
  - **Description**: Runs only the Late-Fusion multimodal network.
  - **Inputs**: `image` (File), 12 atmospheric parameters, `case_id` (String)
  - **Returns**: `FusionPredictionResponse` containing `fusion_signal` (0.0 - 1.0).

- **`POST /scoring/risk-index`**
  - **Description**: Calculates the 0–100 Development/Risk Index using all models.
  - **Inputs**: `image` (File), 12 atmospheric parameters, `case_id` (String)
  - **Returns**: `RiskIndexResponse` containing `development_risk_index` and `components`.

---

## 3. Response Schema & Case Workflows

### 3.1. High-Risk / Verified Cyclone (Case A)
When all per-input signals $\ge 0.50$, the verification gate triggers LLM vision review:

```json
{
  "case_id": "TC-01",
  "prediction": {
    "classification": "cyclone",
    "visual_signal": 0.830,
    "environmental_signal": 0.910,
    "fusion_signal": 0.603,
    "development_risk_index": 77.5,
    "confidence_pct": 60.3,
    "components": {
      "visual": 0.603,
      "environmental": 0.890,
      "convection": 0.880,
      "moisture": 0.880,
      "organization": 0.920,
      "persistence": 0.900
    },
    "data_source_tag": "synthetic_mvp",
    "note": "Development/Risk Index — not a calibrated probability."
  },
  "verification": {
    "gate_triggered": true,
    "llm_called": true,
    "status": "VERIFIED_CYCLONE",
    "diagnostic": "all_signals_above_threshold_and_llm_confirmed",
    "assessment": "cyclone_consistent",
    "confidence": 0.92,
    "reason": "Distinct eye and spiral curved convective banding.",
    "needs_human_review": false
  },
  "explanation": {
    "type": "high_risk_verification",
    "summary": "Multimodal ML models and secondary visual verification confirm active cyclogenesis with a Development/Risk Index of 77.5/100.",
    "atmospheric_rationale": "All per-input prediction matrices exceed the 50% threshold (Visual=0.83, Numerical=0.91, Fusion=0.60). Satellite imagery exhibits pronounced spiral banding and curved convective structure, which is corroborated by favorable atmospheric parameters.",
    "primary_drivers": [
      "Favorable sea surface temperature (30.5°C >= 26.5°C threshold)",
      "Low vertical wind shear (4.5 m/s) permitting convective column retention",
      "Moist mid-troposphere (88% RH) fueling latent heat release",
      "Elevated vorticity / rotation index (0.85)",
      "High cloud organization index (0.92)"
    ],
    "inhibiting_barriers": [
      "No major suppressing barriers detected in ambient field."
    ],
    "operational_guidance": "Issue cyclone warning advisory and initiate standard tropical disturbance tracking protocol."
  },
  "timing_ms": {
    "total_ms": 142.50
  }
}
```

### 3.2. Low-Risk / Suppressed Cyclone (Case B)
When any signal $< 0.50$, the verification gate is bypassed (`LLM_NOT_REQUIRED`), saving API costs while providing detailed atmospheric suppression explanations:

```json
{
  "case_id": "TC-02",
  "prediction": {
    "classification": "non_cyclone",
    "visual_signal": 0.825,
    "environmental_signal": 0.080,
    "fusion_signal": 0.380,
    "development_risk_index": 18.2,
    "confidence_pct": 62.0,
    "components": { ... },
    "data_source_tag": "synthetic_mvp",
    "note": "Development/Risk Index — not a calibrated probability."
  },
  "verification": {
    "gate_triggered": false,
    "llm_called": false,
    "status": "LLM_NOT_REQUIRED",
    "diagnostic": "numerical_probability_below_threshold",
    "assessment": null,
    "confidence": null,
    "reason": null,
    "needs_human_review": null
  },
  "explanation": {
    "type": "suppressed_or_low_risk",
    "summary": "Cyclogenesis is suppressed or absent with a Development/Risk Index of 18.2/100. Secondary LLM verification was not required.",
    "atmospheric_rationale": "One or more prediction matrices fall below the 50% trigger threshold (Numerical Probability (0.08), Fusion Probability (0.38)). Atmospheric environmental conditions or visual cloud structures do not support organized cyclonic development.",
    "primary_drivers": [
      "Low vertical wind shear (12.0 m/s) permitting convective column retention"
    ],
    "inhibiting_barriers": [
      "Sub-optimal ocean heat content / SST (26.0°C < 26.5°C)",
      "Dry air entrainment (35% RH) suppressing convective towers",
      "Weak atmospheric vorticity / rotation index (0.02)",
      "Disorganized cloud canopy (0.05)"
    ],
    "operational_guidance": "Normal meteorological monitoring. No cyclone alert required."
  },
  "timing_ms": {
    "total_ms": 98.30
  }
}
```

---

## 4. Running the Backend Server

Start the development server with hot reloading:

```bash
uvicorn backend.app.main:app --host 0.0.0.0 --port 8000 --reload
```

Interactive OpenAPI documentation will be accessible at:
- **Swagger UI**: [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)
- **ReDoc**: [http://127.0.0.1:8000/redoc](http://127.0.0.1:8000/redoc)

---

## 5. Configuration & Environment Variables

| Variable | Default | Purpose |
| :--- | :--- | :--- |
| `CYCLOSENSE_CORS_ORIGINS` | `http://localhost:3000,http://localhost:5173,...` | Comma-separated allowed frontend origins |
| `CYCLOSENSE_LLM_PROVIDER` | `"mock"` | Vision verifier provider (`"mock"` or `"groq"`) |
| `CYCLOSENSE_LLM_GATE_THRESHOLD` | `0.50` | Trigger threshold for secondary verification |
| `CYCLOSENSE_LLM_CONFIDENCE_THRESHOLD`| `0.70` | Confidence cutoff for `VERIFIED_CYCLONE` |
| `CYCLOSENSE_LLM_EXPLANATION_PROVIDER`| `"mock"` | Explanation generator provider (`"mock"` or `"groq"`) |
| `GROQ_API_KEY` | None | API key for live Groq models (when provider is `"groq"`) |

---

## 6. Error Handling & Status Codes

| Status Code | Reason | Cause |
| :--- | :--- | :--- |
| **`400 Bad Request`** | Invalid Image | Missing image file, unsupported file extension, corrupted file, or non-RGB color mode |
| **`422 Unprocessable Entity`** | Validation Error | Missing numerical form field, non-numeric value, or value outside physical meteorological bounds |
| **`500 Internal Server Error`**| Inference Error | PyTorch tensor shape mismatch or pipeline execution exception (sanitized message returned) |
| **`503 Service Unavailable`** | Configuration Error| Model checkpoints or scaler joblib file missing on disk |

---

## 7. Frontend Integration Example (JavaScript / TypeScript)

```typescript
const formData = new FormData();
formData.append("image", fileInput.files[0]);
formData.append("case_id", "TC-01");
formData.append("temperature_c", "28.5");
formData.append("sea_surface_temperature_c", "30.5");
formData.append("relative_humidity_pct", "88.0");
formData.append("water_vapour_gkg", "22.0");
formData.append("surface_pressure_hpa", "975.0");
formData.append("wind_speed_ms", "42.0");
formData.append("wind_direction_deg", "120.0");
formData.append("vertical_wind_shear_ms", "4.5");
formData.append("cloud_organization_index", "0.92");
formData.append("convection_index", "0.88");
formData.append("rotation_index", "0.85");
formData.append("persistence_index", "0.90");

const response = await fetch("http://127.0.0.1:8000/predict", {
  method: "POST",
  body: formData,
});

if (!response.ok) {
  const err = await response.json();
  console.error("Prediction failed:", err.detail);
} else {
  const result = await response.json();
  console.log("Risk Index:", result.prediction.development_risk_index);
  console.log("Verification:", result.verification.status);
  console.log("Summary:", result.explanation.summary);
}
```
