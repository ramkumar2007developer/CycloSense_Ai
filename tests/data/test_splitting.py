"""Leakage prevention tests."""

import pandas as pd

from src.data.splitting import split_by_group


def test_no_storm_leakage() -> None:
    df = pd.DataFrame(
        {
            "storm_id": [f"S{i//3}" for i in range(30)],
            "value": list(range(30)),
        }
    )
    splits = split_by_group(df, "storm_id", seed=42)
    train_storms = set(splits.train["storm_id"])
    val_storms = set(splits.validation["storm_id"])
    test_storms = set(splits.test["storm_id"])
    assert not (train_storms & val_storms or train_storms & test_storms or val_storms & test_storms)
