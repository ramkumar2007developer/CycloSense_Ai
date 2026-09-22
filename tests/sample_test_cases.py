"""Sample meteorological test cases for CycloSense AI.

This module provides 10 domain-specific scenarios ranging from clear sky and
benign cloud formations to developing depressions and severe cyclones, complete
with actual sample satellite infrared images in sample_data/images/.

Usage:
    - Run as pytest suite:
        pytest tests/sample_test_cases.py -v

    - Run as standalone CLI report:
        python tests/sample_test_cases.py
"""

from __future__ import annotations

import json
from pathlib import Path
import sys
from typing import Any

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pytest
from PIL import Image
import torch
from torchvision import transforms

from src.models.image_cnn import ImageBaselineCNN
from src.scoring.risk_index import ScoringWeights, compute_risk_index

CASES_JSON_PATH = Path(__file__).parent / "sample_test_cases.json"


def load_sample_test_cases() -> list[dict[str, Any]]:
    """Load the 10 sample test case definitions from JSON."""
    with open(CASES_JSON_PATH, encoding="utf-8") as f:
        return json.load(f)


SAMPLE_CASES = load_sample_test_cases()
DEFAULT_WEIGHTS = ScoringWeights(0.4, 0.3, 0.1, 0.1, 0.05, 0.05)


@pytest.mark.parametrize("case", SAMPLE_CASES, ids=[c["id"] for c in SAMPLE_CASES])
def test_sample_image_exists_and_valid(case: dict[str, Any]) -> None:
    """Validate that every sample test case references a valid image on disk."""
    rel_path = case.get("sample_image_path")
    assert rel_path is not None, f"Case {case['id']} missing 'sample_image_path'"

    img_path = PROJECT_ROOT / rel_path
    assert img_path.is_file(), f"Sample image not found: {img_path}"

    # Verify PIL can open and parse image
    with Image.open(img_path) as img:
        img.verify()

    with Image.open(img_path) as img:
        rgb = img.convert("RGB")
        assert rgb.size[0] > 0 and rgb.size[1] > 0


@pytest.mark.parametrize("case", SAMPLE_CASES, ids=[c["id"] for c in SAMPLE_CASES])
def test_sample_case_execution_and_ranges(case: dict[str, Any]) -> None:
    """Validate that each sample case executes cleanly and produces expected risk range."""
    probs = np.array(case["visual_probability"], dtype=float)
    env = case["environmental_features"]

    result = compute_risk_index(
        class_probabilities=probs,
        env_features=env,
        weights=DEFAULT_WEIGHTS,
        data_source_tag="sample_test_cases",
    )

    # Invariants
    assert 0.0 <= result.development_risk_index <= 100.0
    assert result.classification in ("cyclone", "non_cyclone")
    assert result.classification == case["expected_classification"], (
        f"Case {case['id']} expected classification {case['expected_classification']}, got {result.classification}"
    )

    min_expected, max_expected = case["expected_risk_range"]
    assert min_expected <= result.development_risk_index <= max_expected, (
        f"Case {case['id']} ({case['name']}): Score {result.development_risk_index:.1f} "
        f"outside expected range [{min_expected}, {max_expected}]"
    )

    # Component bounds
    for comp_name, comp_val in result.components.as_dict().items():
        assert 0.0 <= comp_val <= 1.0, f"Component {comp_name} out of bounds: {comp_val}"


def test_sample_image_model_inference() -> None:
    """Verify that trained ImageBaselineCNN can perform inference on sample images."""
    model_path = PROJECT_ROOT / "models" / "image_model" / "model.pt"
    if not model_path.is_file():
        pytest.skip("Trained image model not yet present.")

    model = ImageBaselineCNN(num_classes=2, pretrained=False)
    model.load_state_dict(torch.load(model_path, map_location="cpu"))
    model.eval()

    tf = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    for case in SAMPLE_CASES:
        img_path = PROJECT_ROOT / case["sample_image_path"]
        with Image.open(img_path) as img:
            t = tf(img.convert("RGB")).unsqueeze(0)

        with torch.no_grad():
            logits = model(t)
            probs = torch.softmax(logits, dim=1)[0].numpy()

        assert len(probs) == 2
        assert 0.0 <= probs[1] <= 1.0
        assert np.isclose(probs.sum(), 1.0)


def run_sample_evaluation_report() -> None:
    """Print an ASCII table evaluating all sample test cases including their sample images."""
    print("=" * 110)
    print("                         CYCLOSENSE AI — SAMPLE TEST CASE & IMAGE EVALUATION REPORT")
    print("=" * 110)
    header = (
        f"{'ID':<6} | {'Name':<32} | {'Sample Image':<24} | "
        f"{'Class':<12} | {'Risk':<10} | {'Status'}"
    )
    print(header)
    print("-" * 110)

    passed_count = 0
    for case in SAMPLE_CASES:
        probs = np.array(case["visual_probability"], dtype=float)
        env = case["environmental_features"]
        result = compute_risk_index(probs, env, DEFAULT_WEIGHTS, data_source_tag="sample_test_cases")

        min_exp, max_exp = case["expected_risk_range"]
        matches_class = result.classification == case["expected_classification"]
        within_range = min_exp <= result.development_risk_index <= max_exp
        status = "PASS [OK]" if (matches_class and within_range) else "FLAG [!]"

        if matches_class and within_range:
            passed_count += 1

        img_filename = Path(case["sample_image_path"]).name
        print(
            f"{case['id']:<6} | "
            f"{case['name'][:32]:<32} | "
            f"{img_filename:<24} | "
            f"{result.classification:<12} | "
            f"{result.development_risk_index:>5.1f} / 100 | "
            f"{status}"
        )

    print("-" * 110)
    print(
        f"Summary: {passed_count}/{len(SAMPLE_CASES)} scenarios and associated sample images "
        f"validated against domain expectations."
    )
    print("=" * 110)


if __name__ == "__main__":
    run_sample_evaluation_report()
