"""Dataset audit utilities — inspect actual files, invent nothing."""

from __future__ import annotations

import csv
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from PIL import Image

from src.config.ml_config import MLConfig
from src.exceptions import DatasetValidationError

_IMAGE_EXTS = {".jpg", ".jpeg", ".png"}


@dataclass
class DatasetAuditReport:
    source_tag: str
    image_count: int = 0
    image_formats: dict[str, int] = field(default_factory=dict)
    image_dimensions: dict[str, int] = field(default_factory=dict)
    corrupted_images: list[str] = field(default_factory=list)
    duplicate_image_hashes: list[str] = field(default_factory=list)
    numerical_row_count: int = 0
    numerical_columns: list[str] = field(default_factory=list)
    missing_values: dict[str, int] = field(default_factory=dict)
    label_distribution: dict[str, int] = field(default_factory=dict)
    group_count: int | None = None
    duplicate_rows: int = 0
    notes: list[str] = field(default_factory=list)
    ambiguities: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "source_tag": self.source_tag,
            "image_count": self.image_count,
            "image_formats": self.image_formats,
            "image_dimensions_sample": self.image_dimensions,
            "corrupted_images": self.corrupted_images,
            "numerical_row_count": self.numerical_row_count,
            "numerical_columns": self.numerical_columns,
            "missing_values": self.missing_values,
            "label_distribution": self.label_distribution,
            "group_count": self.group_count,
            "duplicate_rows": self.duplicate_rows,
            "notes": self.notes,
            "ambiguities": self.ambiguities,
        }


def _resolve_env_csv(config: MLConfig) -> Path:
    rel = config.raw["paths"]["environmental_csv"]
    candidates = [
        config.path(rel),
        config.project_root / rel,
        Path.home() / "Downloads" / "cyclosense_mvp_environmental_dataset.csv",
    ]
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    raise DatasetValidationError(
        f"synthetic_mvp environmental CSV not found. Tried: {[str(c) for c in candidates]}"
    )


def audit_datasets(config: MLConfig) -> DatasetAuditReport:
    """Audit image and numerical datasets from actual local files."""
    report = DatasetAuditReport(source_tag=config.data_source_tag)
    report.notes.append(
        "Numerical environmental CSV is synthetic_mvp — not real meteorological measurements."
    )

    # Images
    img_dir = config.path(config.raw["paths"]["insat_image_dir"])
    if not img_dir.is_dir():
        raise DatasetValidationError(f"Image directory not found: {img_dir}")

    hashes: Counter[str] = Counter()
    for path in sorted(img_dir.iterdir()):
        if not path.is_file() or path.suffix.lower() not in _IMAGE_EXTS:
            continue
        report.image_count += 1
        report.image_formats[path.suffix.lower()] = (
            report.image_formats.get(path.suffix.lower(), 0) + 1
        )
        try:
            with Image.open(path) as im:
                im.verify()
            with Image.open(path) as im:
                dim = f"{im.size[0]}x{im.size[1]}"
                report.image_dimensions[dim] = report.image_dimensions.get(dim, 0) + 1
        except Exception:
            report.corrupted_images.append(path.name)
            continue
        hashes[path.read_bytes()[:4096].hex()] += 1

    report.duplicate_image_hashes = [h for h, c in hashes.items() if c > 1]

    # INSAT label CSV semantics
    label_csv = config.path(config.raw["paths"]["insat_label_csv"])
    if label_csv.is_file():
        with label_csv.open(encoding="utf-8-sig", newline="") as handle:
            rows = list(csv.DictReader(handle))
        report.notes.append(
            f"INSAT label CSV: {len(rows)} rows; columns img_name/label only."
        )
        report.ambiguities.append(
            "INSAT 'label' column meaning is not documented in source metadata."
        )
    else:
        report.ambiguities.append(f"INSAT label CSV not found: {label_csv}")

    report.ambiguities.append(
        "INSAT infrared folder name suggests cyclone-only images; "
        "non-cyclone cloud scenes may be underrepresented."
    )
    report.ambiguities.append(
        "synthetic_mvp image_id values (IMG_0001.jpg) do not match INSAT filenames (25.jpg)."
    )

    # Numerical CSV
    env_csv = _resolve_env_csv(config)
    with env_csv.open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        report.numerical_columns = list(reader.fieldnames or [])
        rows = list(reader)

    report.numerical_row_count = len(rows)
    report.notes.append(f"Environmental CSV path: {env_csv}")

    for col in report.numerical_columns:
        missing = sum(1 for row in rows if not str(row.get(col, "")).strip())
        if missing:
            report.missing_values[col] = missing

    label_col = config.raw["numerical"]["label_column"]
    if label_col in report.numerical_columns:
        report.label_distribution = dict(Counter(row[label_col] for row in rows))

    group_col = config.raw["numerical"]["group_column"]
    if group_col in report.numerical_columns:
        report.group_count = len({row[group_col] for row in rows})

    keys = [tuple(row[c] for c in report.numerical_columns) for row in rows]
    report.duplicate_rows = len(keys) - len(set(keys))

    return report
