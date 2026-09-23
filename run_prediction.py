"""CycloSense AI — Terminal Prediction, Verification & Explanation Runner.

Usage:
  python run_prediction.py                    # Run comprehensive demo across all representative cases
  python run_prediction.py --case TC-01       # Run specific case by ID (TC-01 to TC-10)
  python run_prediction.py --row 2            # Run specific row from sample_environmental.csv
  python run_prediction.py --image sample_data/images/insat_cyclone_101.jpg --row 0
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd
from src.inference.predictor import CycloSensePredictor
from src.inference.llm_verifier import MockLLMVerifier
from src.inference.llm_explainer import MockLLMExplainer

# Default image mappings for sample cases
CASE_IMAGE_MAP = {
    "TC-01": "sample_data/images/insat_cyclone_101.jpg",
    "TC-02": "sample_data/images/clear_sky_ocean.jpg",
    "TC-03": "sample_data/images/benign_cloud_cluster.jpg",
    "TC-04": "sample_data/images/shear_disorganized_band.jpg",
    "TC-05": "sample_data/images/insat_cyclone_106.jpg",
    "TC-06": "sample_data/images/insat_cyclone_111.jpg",
    "TC-07": "sample_data/images/insat_cyclone_112.jpg",
    "TC-08": "sample_data/images/shear_disorganized_band.jpg",
    "TC-09": "sample_data/images/benign_cloud_cluster.jpg",
    "TC-10": "sample_data/images/clear_sky_ocean.jpg",
}


def print_banner() -> None:
    print("=" * 105)
    print("   CYCLOSENSE AI — MULTIMODAL PREDICTION, SECONDARY VERIFICATION & LLM EXPLANATION")
    print("=" * 105)


def display_result(case_id: str, case_name: str, image_path: Path, env_row: pd.Series, result: dict[str, object]) -> None:
    expl = result.get("llm_explanation", {})

    print(f"\n>> CASE: [{case_id}] {case_name}")
    print(f"   Satellite Image: {image_path.name}")
    print("-" * 105)

    # 1. Numerical Atmospheric Inputs
    print("   [1] NUMERICAL ATMOSPHERIC FEATURES:")
    print(
        f"       SST:          {float(env_row['sea_surface_temperature_c']):>5.1f} °C   | "
        f"Wind Shear:     {float(env_row['vertical_wind_shear_ms']):>5.1f} m/s | "
        f"Wind Speed:   {float(env_row['wind_speed_ms']):>5.1f} m/s"
    )
    print(
        f"       Humidity:     {float(env_row['relative_humidity_pct']):>5.1f} %    | "
        f"Pressure:     {float(env_row['surface_pressure_hpa']):>6.1f} hPa | "
        f"Organization: {float(env_row['cloud_organization_index']):>5.2f}"
    )
    print(
        f"       Vorticity/Rot:{float(env_row['rotation_index']):>5.2f}       | "
        f"Convection:     {float(env_row['convection_index']):>5.2f}     | "
        f"Persistence:  {float(env_row['persistence_index']):>5.2f}"
    )

    # 2. Prediction Matrices
    print("\n   [2] ML PREDICTION MATRICES & PROBABILITIES:")
    img_p = float(result.get("image_probability", 0.0))
    num_p = float(result.get("numerical_probability", 0.0))
    fus_p = float(result.get("fusion_probability", 0.0))
    risk_idx = float(result.get("development_risk_index", 0.0))
    cls_name = str(result.get("classification", "unknown")).upper()
    conf_pct = float(result.get("confidence_pct", 0.0))

    print(f"       Visual Signal (Image CNN):     {img_p:.3f} ({img_p * 100:.1f}%)")
    print(f"       Environmental Signal (MLP):    {num_p:.3f} ({num_p * 100:.1f}%)")
    print(f"       Multimodal Fusion Signal:      {fus_p:.3f} ({fus_p * 100:.1f}%)")
    print(f"       -------------------------------------------------------------")
    print(f"       CLASSIFICATION: {cls_name:<12} (Confidence: {conf_pct:.1f}%)")
    print(f"       DEVELOPMENT/RISK INDEX:       {risk_idx:>5.1f} / 100")

    # 3. Verification Gate
    all_50 = bool(result.get("all_signals_above_50", False))
    llm_called = bool(result.get("llm_called", False))
    status = str(result.get("final_status", "LLM_NOT_REQUIRED"))
    reason = str(result.get("reason", ""))

    print("\n   [3] SECONDARY VERIFICATION GATE:")
    print(f"       Gate Triggered (All Signals >= 50%): {all_50}")
    print(f"       LLM Vision Review Called:            {llm_called}")
    print(f"       Verification Status:                 [ {status} ]")
    if reason:
        print(f"       Gate Diagnostic:                     {reason}")

    # 4. LLM Meteorological Explanation
    if expl:
        print("\n   [4] LLM METEOROLOGICAL EXPLANATION:")
        print(f"       Type: {expl.get('case_type', 'standard')}")
        print(f"       Summary:")
        print(f"         \"{expl.get('summary', '')}\"")
        print(f"       Atmospheric Rationale:")
        print(f"         {expl.get('meteorological_rationale', '')}")
        print(f"       Primary Drivers:")
        for d in expl.get("primary_drivers", []):
            print(f"         + {d}")
        print(f"       Inhibiting Barriers:")
        for inh in expl.get("inhibiting_factors", []):
            print(f"         - {inh}")
        print(f"       Operational Guidance:")
        print(f"         >>> {expl.get('operational_guidance', '')} <<<")

    print("-" * 105)


def run_single(predictor: CycloSensePredictor, env_df: pd.DataFrame, idx: int, image_override: Path | None = None) -> None:
    row = env_df.iloc[idx]
    cid = str(row.get("case_id", f"ROW-{idx}"))
    cname = str(row.get("case_name", "Observation"))

    if image_override and image_override.is_file():
        img_path = image_override
    else:
        mapped = CASE_IMAGE_MAP.get(cid, "sample_data/images/insat_cyclone_101.jpg")
        img_path = PROJECT_ROOT / mapped

    # Configure mock verifier behavior according to ground truth case type for realistic testing
    if cid in ("TC-01", "TC-06", "TC-07"):
        verifier = MockLLMVerifier(response="cyclone_consistent", confidence=0.92, reason="Distinct eye and spiral curved convective banding.")
    elif cid in ("TC-03", "TC-05"):
        verifier = MockLLMVerifier(response="non_cyclone_consistent", confidence=0.88, reason="Benign cloud cluster / unorganized convection.")
    else:
        verifier = MockLLMVerifier(response="non_cyclone_consistent", confidence=0.85, reason="Disorganized clouds without cyclonic circulation.")

    result = predictor.predict_with_explanation(
        image_path=img_path,
        env_row=row,
        verifier=verifier,
        image_id=cid,
    )

    display_result(cid, cname, img_path, row, result)


def main() -> None:
    parser = argparse.ArgumentParser(description="CycloSense AI Prediction & Explanation Terminal Runner")
    parser.add_argument("--case", type=str, default=None, help="Case ID to run (e.g. TC-01, TC-03, TC-05)")
    parser.add_argument("--row", type=int, default=None, help="Row index from environmental CSV (0 to 9)")
    parser.add_argument("--env", type=str, default="sample_data/sample_environmental.csv", help="Path to numerical CSV")
    parser.add_argument("--image", type=str, default=None, help="Optional image path override")
    args = parser.parse_args()

    print_banner()

    env_path = PROJECT_ROOT / args.env
    if not env_path.is_file():
        print(f"Error: Environmental CSV not found at {env_path}")
        sys.exit(1)

    env_df = pd.read_csv(env_path)
    predictor = CycloSensePredictor.from_project()

    img_override = Path(args.image).resolve() if args.image else None

    if args.case:
        matches = env_df[env_df["case_id"] == args.case]
        if matches.empty:
            print(f"Error: Case ID {args.case} not found in {env_path.name}")
            sys.exit(1)
        run_single(predictor, env_df, matches.index[0], img_override)
    elif args.row is not None:
        if args.row < 0 or args.row >= len(env_df):
            print(f"Error: Row index {args.row} out of range (0 to {len(env_df) - 1})")
            sys.exit(1)
        run_single(predictor, env_df, args.row, img_override)
    else:
        # Showcase 4 distinct representative cases:
        # TC-01 (Active cyclone, Case A verified)
        # TC-02 (Calm ocean, Case B low risk)
        # TC-03 (Benign cloud cluster, Anti-bias check)
        # TC-05 (Dry air intrusion, Case A conflict)
        showcase_cases = ["TC-01", "TC-02", "TC-03", "TC-05"]
        print(f"Running showcase across representative meteorological cases: {', '.join(showcase_cases)}")
        for cid in showcase_cases:
            matches = env_df[env_df["case_id"] == cid]
            if not matches.empty:
                run_single(predictor, env_df, matches.index[0])

    print("\nTerminal prediction and evaluation completed successfully!")


if __name__ == "__main__":
    main()
