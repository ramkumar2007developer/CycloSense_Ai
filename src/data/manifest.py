"""Image ↔ numerical pairing manifest (synthetic_mvp pairing)."""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from src.config.ml_config import MLConfig
from src.data.audit import _resolve_env_csv
from src.exceptions import DatasetValidationError

PAIRING_TAG = "synthetic_mvp_pairing"


@dataclass(frozen=True)
class ManifestRow:
    sample_id: str
    image_path: Path
    image_filename: str
    storm_id: str
    timestamp: str
    cyclone_label: int
    data_source_tag: str
    pairing_tag: str
    insat_group_label: str


def build_synthetic_mvp_manifest(config: MLConfig) -> pd.DataFrame:
    """Pair INSAT images with synthetic_mvp environmental rows for pipeline validation.

    IMPORTANT: This is synthetic_mvp_pairing — environmental values were NOT
    physically measured for these satellite images. Used only to validate the
    software/ML pipeline.
    """
    img_dir = config.path(config.raw["paths"]["insat_image_dir"])
    env_csv = _resolve_env_csv(config)
    env_df = pd.read_csv(env_csv)

    image_files = sorted(
        p for p in img_dir.iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png"}
    )
    if not image_files:
        raise DatasetValidationError(f"No images found in {img_dir}")

    label_csv = config.path(config.raw["paths"]["insat_label_csv"])
    insat_labels: dict[str, str] = {}
    if label_csv.is_file():
        with label_csv.open(encoding="utf-8-sig", newline="") as handle:
            for row in csv.DictReader(handle):
                insat_labels[row["img_name"]] = row["label"]

    rows: list[dict[str, object]] = []
    for index, image_path in enumerate(image_files):
        env_row = env_df.iloc[index % len(env_df)]
        rows.append(
            {
                "sample_id": f"pair_{index:04d}",
                "image_path": str(image_path.resolve()),
                "image_filename": image_path.name,
                "storm_id": str(env_row["storm_id"]),
                "timestamp": str(env_row["timestamp"]),
                "cyclone_label": int(env_row["cyclone_label"]),
                "data_source_tag": config.data_source_tag,
                "pairing_tag": PAIRING_TAG,
                "insat_group_label": insat_labels.get(image_path.name, "Not available in source dataset."),
            }
        )

    manifest = pd.DataFrame(rows)
    return manifest


def save_manifest(manifest: pd.DataFrame, output_path: Path) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    manifest.to_csv(output_path, index=False)
    return output_path
