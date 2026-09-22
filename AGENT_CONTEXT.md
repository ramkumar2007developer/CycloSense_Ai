# Agent Context — CycloSense AI

**Phase:** ML/DL pipeline (no API)

## Rules

- Numerical CSV = `synthetic_mvp` (not real observations)
- Pairing = `synthetic_mvp_pairing` (not physical measurement pairing)
- Do not implement FastAPI until ML validation passes
- Read `AGENT_ACTION_LOG.txt` before resuming work

## Data locations

- Images: `%CYCLOSENSE_EXTERNAL_DATA_ROOT%/archive/insat3d_ir_cyclone_ds/CYCLONE_DATASET_INFRARED`
- Env CSV: `~/Downloads/cyclosense_mvp_environmental_dataset.csv`
