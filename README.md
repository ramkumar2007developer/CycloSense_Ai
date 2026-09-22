# CycloSense AI

AI-assisted tropical cyclone monitoring research/MVP prototype — **ML/DL pipeline only** (no API yet).

## Setup

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
set CYCLOSENSE_EXTERNAL_DATA_ROOT=C:\Users\Hp\Downloads\dataset-main\dataset-main
```

Environmental CSV (synthetic_mvp): `Downloads/cyclosense_mvp_environmental_dataset.csv`

## Run tests

```bash
pytest -q
pytest -m slow   # includes 1-epoch training integration
```

## Run ML pipeline

```bash
python -m src.training.run_ml_pipeline
```

## Important

- Numerical environmental data is **synthetic_mvp** — not real meteorological measurements.
- Image↔numerical pairing uses **synthetic_mvp_pairing** for pipeline validation only.
