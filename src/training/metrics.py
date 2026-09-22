"""Evaluation metrics — computed from actual predictions only."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


@dataclass(frozen=True)
class MetricReport:
    accuracy: float
    precision: float
    recall: float
    f1: float
    roc_auc: float | None
    confusion: list[list[int]]
    loss: float | None = None
    single_class_warning: bool = False
    """True when y_true contains only one class — metrics are unreliable.

    This is a known limitation of the synthetic_mvp dataset where the
    87.7% class-1 imbalance combined with small split sizes can produce
    validation/test folds with only cyclone (class-1) samples.
    """

    def as_dict(self) -> dict[str, object]:
        d: dict[str, object] = {
            "accuracy": self.accuracy,
            "precision": self.precision,
            "recall": self.recall,
            "f1": self.f1,
            "roc_auc": self.roc_auc,
            "confusion_matrix": self.confusion,
            "loss": self.loss,
        }
        if self.single_class_warning:
            d["WARNING"] = (
                "single_class_split: only one class present in this split. "
                "Accuracy/precision/recall are unreliable. "
                "Root cause: synthetic_mvp class imbalance (87.7% cyclone-1)."
            )
        return d


def compute_metrics(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    loss: float | None = None,
) -> MetricReport:
    """Compute classification metrics from actual predictions.

    Args:
        y_true: Ground-truth integer labels (0 or 1).
        y_prob: Predicted probability for class 1.
        loss: Optional pre-computed loss value.

    Returns:
        MetricReport with all metrics. single_class_warning=True when
        y_true has only one unique class (metrics unreliable).
    """
    n_classes = len(np.unique(y_true))
    single_class = n_classes < 2

    y_pred = (y_prob >= 0.5).astype(int)
    roc: float | None
    try:
        roc = float(roc_auc_score(y_true, y_prob)) if not single_class else None
    except ValueError:
        roc = None

    return MetricReport(
        accuracy=float(accuracy_score(y_true, y_pred)),
        precision=float(precision_score(y_true, y_pred, zero_division=0)),
        recall=float(recall_score(y_true, y_pred, zero_division=0)),
        f1=float(f1_score(y_true, y_pred, zero_division=0)),
        roc_auc=roc,
        confusion=confusion_matrix(y_true, y_pred, labels=[0, 1]).tolist(),
        loss=loss,
        single_class_warning=single_class,
    )
