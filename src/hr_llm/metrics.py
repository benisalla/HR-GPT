"""Task metrics. Accuracy alone is misleading here (84% of employees stay), so we also report
balanced accuracy and macro-F1 for every classifier, ROC-AUC for attrition, and MAE / R² for income.
"""

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    mean_absolute_error,
    r2_score,
    roc_auc_score,
)

from hr_llm.config import CLASSIFICATION, TaskSpec

# The one number per task used in summaries ("higher is better" for all of them)
HEADLINE = {CLASSIFICATION: "balanced_accuracy", "regression": "r2"}


def classification_metrics(y_true: np.ndarray, probs: np.ndarray) -> dict[str, float]:
    """y_true [N] class ids, probs [N, C] -> accuracy, balanced accuracy, macro-F1 (+ AUC if C=2)."""
    pred = probs.argmax(axis=1)
    labels = list(range(probs.shape[1]))
    out = {
        "accuracy": accuracy_score(y_true, pred),
        "balanced_accuracy": balanced_accuracy_score(y_true, pred),
        "macro_f1": f1_score(y_true, pred, labels=labels, average="macro", zero_division=0),
    }
    if probs.shape[1] == 2 and len(np.unique(y_true)) == 2:
        out["roc_auc"] = roc_auc_score(y_true, probs[:, 1])
        out["f1_positive"] = f1_score(y_true, pred, pos_label=1, zero_division=0)
    return {k: float(v) for k, v in out.items()}


def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    """In the target's own units (e.g. dollars per month)."""
    return {
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "rmse": float(np.sqrt(np.mean((y_true - y_pred) ** 2))),
        "r2": float(r2_score(y_true, y_pred)),
    }


def task_metrics(spec: TaskSpec, y_true: np.ndarray, prediction: np.ndarray) -> dict[str, float]:
    if spec.kind == CLASSIFICATION:
        return classification_metrics(y_true, prediction)
    return regression_metrics(y_true, prediction)


def headline(spec: TaskSpec, metrics: dict[str, float]) -> float:
    return metrics[HEADLINE[spec.kind]]
