# CycloSense AI — Backend Deployment Guide on Render

This guide provides step-by-step instructions to deploy the CycloSense FastAPI backend service to **Render.com**.

---

## 1. Pre-Deployment Checklist (Files & Git)

Before deploying, ensure the required backend files, dependencies, and model weights are committed to your Git repository:

1. **Model Weights & Preprocessors**:
   The model files (~12.2 MB total) must be tracked in Git so Render can load them at startup:
   - `models/image_model/model.pt`
   - `models/numerical_model/model.pt`
   - `models/fusion_model/model.pt`
   - `artifacts/preprocessing/numerical_scaler.joblib`

2. **Backend Dependencies (`requirements.txt`)**:
   FastAPI, Uvicorn, Pydantic, Pillow, PyTorch, Scikit-learn, etc., are configured in [requirements.txt](file:///c:/Users/Hp/OneDrive/Desktop/CycloSense/requirements.txt).

3. **Stage & Commit Git Changes**:
   ```bash
   git add requirements.txt .gitignore render.yaml backend/ models/ artifacts/preprocessing/ docs/
   git commit -m "chore: prepare repository for Render backend deployment"
   git push origin main
   ```

---

## 2. Render Deployment Options

You can deploy using either the **Render Web Dashboard** (recommended for beginners) or **Render Blueprint** (`render.yaml`).

### Option A: Render Dashboard (Step-by-Step UI)

1. Go to [https://dashboard.render.com](https://dashboard.render.com) and log in.
2. Click **New +** in the top right and select **Web Service**.
3. Choose **Build and deploy from a Git repository** and connect your GitHub/GitLab account.
4. Select your **`CycloSense`** repository.
5. Fill in the deployment settings:

| Setting Field | Value | Notes |
| :--- | :--- | :--- |
| **Name** | `cyclosense-backend` | Or any unique name for your API URL |
| **Region** | Select closest (e.g. *Oregon (US West)* or *Frankfurt (EU)*) | Low latency to your target users |
| **Branch** | `main` | Production branch |
| **Root Directory** | *(Leave blank)* | Uses root of repo |
| **Runtime** | `Python 3` | Native Python runtime |
| **Build Command** | `pip install --upgrade pip && pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu && pip install -r requirements.txt` | **Critical:** Installing CPU PyTorch saves ~2.3 GB of CUDA bloat and prevents build memory timeouts. |
| **Start Command** | `python -m uvicorn backend.app.main:app --host 0.0.0.0 --port $PORT` | Uses `python -m uvicorn` to guarantee finding uvicorn in Render's Python path |
| **Instance Type** | `Free` (or `Starter`) | 512MB RAM free tier handles CPU inference cleanly |

6. Scroll down to **Advanced** settings:
   - **Health Check Path**: `/health`
   - **Auto-Deploy**: `Yes` (deploys on every `git push origin main`)

7. Add **Environment Variables** (under *Environment Variables* section):

| Key | Example Value | Description |
| :--- | :--- | :--- |
| `PYTHON_VERSION` | `3.11.9` | Ensures a stable Python 3.11 build environment |
| `CYCLOSENSE_CORS_ORIGINS` | `http://localhost:5173,https://your-frontend.vercel.app` | Comma-separated allowed frontend origins for CORS |
| `GROQ_API_KEY` | `gsk_...` *(Required for live LLM)* | Your Groq Cloud API Key for live Llama-3.3-70B explanations and Llama-3.2-11B vision verification |

> [!IMPORTANT]
> **How to get your free `GROQ_API_KEY` (Takes 30 seconds, 100% Free):**
> 1. Go to **[https://console.groq.com/keys](https://console.groq.com/keys)**
> 2. Sign up / log in with your Google or GitHub account (no credit card required).
> 3. Click **Create API Key**, give it a name (e.g. `CycloSense-Production`), and copy the generated key (`gsk_...`).
> 4. In Render, paste this key into the `GROQ_API_KEY` environment variable.

8. Click **Create Web Service**.

---

### Option B: Render Blueprint (`render.yaml`)

A pre-configured [render.yaml](file:///c:/Users/Hp/OneDrive/Desktop/CycloSense/render.yaml) is provided in the repository root:

1. In Render Dashboard, click **New +** -> **Blueprint**.
2. Connect your Git repository.
3. Render reads `render.yaml` automatically and configures the service, build command, health check, and environment variables.
4. Click **Apply**.

---

## 3. Verifying the Deployment

Once Render finishes building and the status turns green (**Live**):

1. **Service URL**:
   Your API will be accessible at:
   `https://<your-service-name>.onrender.com`

2. **Health Check**:
   Open in your browser:
   ```
   https://<your-service-name>.onrender.com/health
   ```
   Expected response:
   ```json
   {
     "status": "ok",
     "service": "CycloSense AI",
     "version": "0.1.0"
   }
   ```

3. **Interactive Swagger Documentation**:
   Visit:
   ```
   https://<your-service-name>.onrender.com/docs
   ```
   You can test endpoints directly from the browser:
   - `POST /predict` (Full multimodal inference)
   - `POST /predict/image` (Satellite CNN only)
   - `POST /predict/numerical` (Atmospheric MLP only)
   - `POST /predict/fusion` (Late-fusion only)
   - `POST /calculate-risk-index` (0-100 risk scoring only)

---

## 4. Free Tier Caveats & Best Practices

- **Spin-down on Inactivity**: Free instances spin down after 15 minutes of zero traffic. The next incoming request will experience a "cold start" of ~30–50 seconds while the container initializes and loads the PyTorch models.
- **CORS Configuration**: Whenever you deploy your frontend (e.g. on Vercel, Netlify, or GitHub Pages), update the `CYCLOSENSE_CORS_ORIGINS` variable in Render to include your frontend URL.
- **Logs**: If anything fails during startup, view the real-time build and application logs under the **Logs** tab in Render.
