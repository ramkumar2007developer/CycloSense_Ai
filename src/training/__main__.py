"""CLI entry: python -m src.training.run_ml_pipeline"""

from src.training.run_ml_pipeline import run_ml_pipeline

if __name__ == "__main__":
    artifacts = run_ml_pipeline()
    print("Pipeline complete. Metrics:", artifacts.metrics_path)
