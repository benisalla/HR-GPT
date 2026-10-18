"""`hr-evaluate`: score a checkpoint on the test split and compare it with two baselines.

Baselines use the same 27 input columns as the model (never the other targets):
- trivial:  always the most frequent class / the median income
- linear:   logistic regression / ridge regression on one-hot + standardised columns

    hr-evaluate --checkpoint checkpoints/run/best.pt
    -> prints a table, writes test_results.json and test_results.md next to the checkpoint
"""

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from hr_llm.config import CLASSIFICATION, TASKS, DataConfig, TaskSpec
from hr_llm.data import frame_to_records, load_split
from hr_llm.features import FEATURES
from hr_llm.metrics import task_metrics
from hr_llm.predict import Predictor

CATEGORICAL = [
    f.name for f in FEATURES if f.is_categorical and isinstance(next(iter(f.choices)), str)
]
NUMERIC = [f.name for f in FEATURES if f.name not in CATEGORICAL]


def true_values(spec: TaskSpec, frame: pd.DataFrame) -> np.ndarray:
    """Class ids (classification) or raw values (regression)."""
    if spec.kind == CLASSIFICATION:
        return frame[spec.column].map(lambda v: spec.values.index(v)).to_numpy()
    return frame[spec.column].astype(float).to_numpy()


def trivial_baseline(spec: TaskSpec, train: pd.DataFrame, test: pd.DataFrame) -> np.ndarray:
    y = true_values(spec, train)
    if spec.kind == CLASSIFICATION:
        prior = np.bincount(y, minlength=len(spec.classes)) / len(y)
        return np.tile(prior, (len(test), 1))  # argmax = majority class; AUC = 0.5
    return np.full(len(test), np.median(y))


def linear_baseline(spec: TaskSpec, train: pd.DataFrame, test: pd.DataFrame) -> np.ndarray:
    columns = ColumnTransformer(
        [
            ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL),
            ("num", StandardScaler(), NUMERIC),
        ]
    )
    y = true_values(spec, train)
    if spec.kind == CLASSIFICATION:
        clf = LogisticRegression(max_iter=5000, class_weight="balanced" if spec.balanced else None)
        model = make_pipeline(columns, clf).fit(train, y)
        return model.predict_proba(test)
    target = np.log(y) if spec.log_target else y
    model = make_pipeline(columns, Ridge(alpha=1.0)).fit(train, target)
    pred = model.predict(test)
    return np.exp(pred) if spec.log_target else pred


def evaluate_checkpoint(
    checkpoint: str | Path, data_config: DataConfig, split: str = "test"
) -> dict[str, Any]:
    train = load_split(data_config, "train")
    test = load_split(data_config, split)
    predictor = Predictor.from_checkpoint(checkpoint)
    model_preds = predictor.predict_many(frame_to_records(test))
    tasks = predictor.tasks or TASKS
    results: dict[str, Any] = {"split": split, "rows": len(test), "checkpoint": str(checkpoint)}
    for name, spec in tasks.items():
        y = true_values(spec, test)
        results[name] = {
            "HR-LLM": task_metrics(spec, y, model_preds[name]),
            "linear": task_metrics(spec, y, linear_baseline(spec, train, test)),
            "trivial": task_metrics(spec, y, trivial_baseline(spec, train, test)),
        }
    return results


ROWS = {
    CLASSIFICATION: [
        ("accuracy", "accuracy", "{:.1%}"),
        ("balanced_accuracy", "balanced acc.", "{:.1%}"),
        ("macro_f1", "macro-F1", "{:.3f}"),
        ("roc_auc", "ROC-AUC", "{:.3f}"),
    ],
    "regression": [
        ("mae", "MAE", "${:,.0f}"),
        ("rmse", "RMSE", "${:,.0f}"),
        ("r2", "R²", "{:.3f}"),
    ],
}


def to_markdown(results: dict[str, Any]) -> str:
    lines = [f"Test split: {results['rows']} employees never seen in training.\n"]
    for name, spec in TASKS.items():
        if name not in results:
            continue
        lines += [
            f"**{name}**\n",
            "| metric | HR-LLM | linear baseline | trivial baseline |",
            "|---|---:|---:|---:|",
        ]
        for key, label, fmt in ROWS[spec.kind]:
            if key not in results[name]["HR-LLM"]:
                continue
            cells = [
                fmt.format(results[name][m][key]) if key in results[name][m] else "-"
                for m in ("HR-LLM", "linear", "trivial")
            ]
            lines.append(f"| {label} | " + " | ".join(cells) + " |")
        lines.append("")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Evaluate HR-LLM on the test split vs baselines.")
    parser.add_argument("--checkpoint", default="checkpoints/run/best.pt")
    parser.add_argument("--data-dir", type=Path, default=DataConfig().data_dir)
    parser.add_argument("--split", default="test", choices=["val", "test"])
    args = parser.parse_args(argv)

    results = evaluate_checkpoint(args.checkpoint, DataConfig(data_dir=args.data_dir), args.split)
    out = Path(args.checkpoint).parent
    (out / f"{args.split}_results.json").write_text(json.dumps(results, indent=2))
    markdown = to_markdown(results)
    (out / f"{args.split}_results.md").write_text(markdown)
    print(markdown)
    print(f"saved {out / f'{args.split}_results.json'} and .md")


if __name__ == "__main__":
    main()
