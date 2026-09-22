"""Storm/group-aware data splitting."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from src.exceptions import ValidationError
from src.utils.reproducibility import set_random_seed


@dataclass(frozen=True)
class SplitIndices:
    train: pd.DataFrame
    validation: pd.DataFrame
    test: pd.DataFrame


def split_by_group(
    df: pd.DataFrame,
    group_column: str,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    seed: int = 42,
) -> SplitIndices:
    """Split dataframe by group IDs to prevent storm leakage."""
    if abs(train_ratio + val_ratio + test_ratio - 1.0) > 1e-6:
        raise ValidationError("Split ratios must sum to 1.0")

    groups = sorted(df[group_column].astype(str).unique())
    if not groups:
        raise ValidationError("No groups found for splitting.")

    set_random_seed(seed)
    import random

    shuffled = groups[:]
    random.shuffle(shuffled)

    n = len(shuffled)
    n_train = max(1, int(n * train_ratio))
    n_val = max(1, int(n * val_ratio)) if n >= 3 else (1 if n > 1 else 0)
    n_test = n - n_train - n_val
    if n_test <= 0 and n >= 3:
        n_test = 1
        n_train = max(1, n_train - 1)

    train_groups = set(shuffled[:n_train])
    val_groups = set(shuffled[n_train : n_train + n_val])
    test_groups = set(shuffled[n_train + n_val :])

    overlap = (train_groups & val_groups) | (train_groups & test_groups) | (val_groups & test_groups)
    if overlap:
        raise ValidationError(f"Group leakage detected: {overlap}")

    return SplitIndices(
        train=df[df[group_column].astype(str).isin(train_groups)].copy(),
        validation=df[df[group_column].astype(str).isin(val_groups)].copy(),
        test=df[df[group_column].astype(str).isin(test_groups)].copy(),
    )
