"""Turn a finished training run into the figures, tables and examples shown in the docs.

    python scripts/make_report.py --checkpoint checkpoints/run/best.pt --name gpt2

Reads `history.json` next to the checkpoint, scores the test split (model + baselines),
and writes:

    docs/images/<name>_training_curves.png   train/val loss and validation metrics per epoch
    docs/images/<name>_income.png            predicted vs actual monthly income (test split)
    docs/examples/<name>_test_predictions.csv  every test employee: inputs, truth, predictions
    docs/examples/<name>_report.md           results tables, confusion matrices, worked examples
"""

import argparse
import json
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from hr_jev.config import CLASSIFICATION, TASKS, DataConfig
from hr_jev.data import frame_to_records, load_split
from hr_jev.evaluate import evaluate_checkpoint, to_markdown
from hr_jev.features import TARGET_COLUMNS, record_to_text
from hr_jev.predict import Predictor

BLUE, ORANGE, AQUA, YELLOW = "#2a78d6", "#eb6834", "#1baf7a", "#eda100"
INK, MUTED, GRID = "#0b0b0b", "#898781", "#e1e0d9"
HEADLINE_LABEL = {
    "Attrition": "Attrition · balanced acc.",
    "JobLevel": "JobLevel · balanced acc.",
    "JobSatisfaction": "JobSatisfaction · balanced acc.",
    "MonthlyIncome": "MonthlyIncome · R²",
}


def style(ax, title):
    ax.set_title(title, loc="left", color=INK, fontsize=11, fontweight="bold")
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color("#c3c2b7")
    ax.tick_params(colors=MUTED, labelsize=9)
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)


def plot_curves(history: list[dict], best_epoch: int, path: Path, name: str) -> None:
    epochs = [h["epoch"] for h in history]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), facecolor="#fcfcfb")
    ax = axes[0]
    ax.plot(epochs, [h["train_loss"] for h in history], color=BLUE, lw=2, marker="o", ms=5, label="train")
    ax.plot(epochs, [h["val_loss"] for h in history], color=ORANGE, lw=2, marker="o", ms=5, label="validation")
    ax.axvline(best_epoch, color=MUTED, lw=1, ls="--")
    ax.text(best_epoch, ax.get_ylim()[1], f" best epoch {best_epoch}", color=MUTED, fontsize=9, va="top")
    ax.set_xlabel("epoch", color=MUTED)
    ax.legend(frameon=False, fontsize=9)
    style(ax, f"Loss (sum of the 4 tasks) · {name}")

    ax = axes[1]
    for color, (task, label) in zip((BLUE, ORANGE, AQUA, YELLOW), HEADLINE_LABEL.items()):
        ax.plot(epochs, [h["val_headline"][task] for h in history], color=color, lw=2, marker="o", ms=5, label=label)
    ax.axhline(0, color="#c3c2b7", lw=1)
    ax.set_xlabel("epoch", color=MUTED)
    ax.set_ylim(min(-0.05, ax.get_ylim()[0]), 1.0)
    ax.legend(frameon=False, fontsize=8, loc="lower right")
    style(ax, "Validation metrics per epoch (higher is better)")
    for a in axes:
        a.set_xticks(epochs)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_income(y_true: np.ndarray, y_pred: np.ndarray, mae: float, r2: float, path: Path, name: str) -> None:
    fig, ax = plt.subplots(figsize=(5.5, 5), facecolor="#fcfcfb")
    lim = (0, max(y_true.max(), y_pred.max()) * 1.05)
    ax.plot(lim, lim, color="#c3c2b7", lw=1, ls="--", label="perfect prediction")
    ax.scatter(y_true, y_pred, s=22, color=BLUE, alpha=0.75, edgecolor="#fcfcfb", linewidth=0.5, label="test employee")
    ax.set_xlim(lim)
    ax.set_ylim(lim)
    ax.set_xlabel("actual monthly income ($)", color=MUTED)
    ax.set_ylabel("predicted monthly income ($)", color=MUTED)
    ax.text(0.03, 0.97, f"MAE ${mae:,.0f} · R² {r2:.3f}", transform=ax.transAxes, va="top", color=INK, fontsize=10)
    ax.legend(frameon=False, fontsize=9, loc="lower right")
    style(ax, f"Monthly income on the 147 test employees · {name}")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def confusion_markdown(spec, y_true: np.ndarray, y_pred: np.ndarray) -> str:
    classes = list(spec.classes)
    k = len(classes)
    m = np.zeros((k, k), dtype=int)
    for t, p in zip(y_true, y_pred):
        m[t, p] += 1
    lines = ["| actual \\ predicted | " + " | ".join(classes) + " |", "|---|" + "---:|" * k]
    for i, c in enumerate(classes):
        cells = [f"**{m[i, j]}**" if i == j else str(m[i, j]) for j in range(k)]
        lines.append(f"| {c} | " + " | ".join(cells) + " |")
    return "\n".join(lines)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", default="checkpoints/run/best.pt")
    p.add_argument("--name", default="gpt2", help="label used in file names and titles")
    p.add_argument("--docs", type=Path, default=Path("docs"))
    p.add_argument("--hardware", default="", help="e.g. 'a Kaggle T4 GPU' or '8-core CPU'")
    p.add_argument("--data-dir", type=Path, default=DataConfig().data_dir)
    args = p.parse_args()

    ckpt = Path(args.checkpoint)
    history = json.loads((ckpt.parent / "history.json").read_text())
    data_config = DataConfig(data_dir=args.data_dir)
    images, examples = args.docs / "images", args.docs / "examples"
    images.mkdir(parents=True, exist_ok=True)
    examples.mkdir(parents=True, exist_ok=True)

    predictor = Predictor.from_checkpoint(ckpt, device="cpu")
    best_epoch = predictor.info["epoch"]
    plot_curves(history, best_epoch, images / f"{args.name}_training_curves.png", args.name)

    # test split: predictions for every employee
    test = load_split(data_config, "test")
    records = frame_to_records(test)
    raw = predictor.predict_many(records)
    results = evaluate_checkpoint(ckpt, data_config)

    table = pd.DataFrame(records)
    truth_idx, pred_idx = {}, {}
    for name, spec in TASKS.items():
        if spec.kind == CLASSIFICATION:
            truth_idx[name] = np.array([spec.values.index(v) for v in test[spec.column]])
            pred_idx[name] = raw[name].argmax(axis=1)
            table[f"pred_{name}"] = [spec.classes[i] for i in pred_idx[name]]
            table[f"prob_{name}"] = raw[name].max(axis=1).round(3)
            table[f"true_{name}"] = [spec.classes[i] for i in truth_idx[name]]
            table[f"hit_{name}"] = pred_idx[name] == truth_idx[name]
        else:
            table[f"pred_{name}"] = raw[name].round(0)
            table[f"true_{name}"] = test[spec.column].to_numpy()
            table[f"abs_error_{name}"] = np.abs(raw[name] - test[spec.column].to_numpy()).round(0)
    table.insert(0, "test_row", range(len(table)))
    table["sentence"] = [record_to_text({k: v for k, v in r.items() if k not in TARGET_COLUMNS}) for r in records]
    table.to_csv(examples / f"{args.name}_test_predictions.csv", index=False)

    y_inc = test["MonthlyIncome"].to_numpy(dtype=float)
    inc = results["MonthlyIncome"]["HR-Jev"]
    plot_income(y_inc, raw["MonthlyIncome"], inc["mae"], inc["r2"], images / f"{args.name}_income.png", args.name)

    # worked examples: 4 where every classifier is right, 4 attrition misses, 2 worst incomes
    hits = table[["hit_Attrition", "hit_JobLevel", "hit_JobSatisfaction"]].sum(axis=1)
    good = table[(hits == 3) & (table["abs_error_MonthlyIncome"] < 800)].head(4)
    leaver_hits = table[(table["true_Attrition"] == "Leaves") & table["hit_Attrition"]].head(2)
    att_miss = table[~table["hit_Attrition"]].sort_values("prob_Attrition", ascending=False).head(4)
    inc_miss = table.sort_values("abs_error_MonthlyIncome", ascending=False).head(2)

    def example(row) -> str:
        lines = [
            f"**Test employee #{row['test_row']}** · what the model reads:",
            "",
            f"> {row['sentence']}",
            "",
            "| Task | Prediction | Actual | |",
            "|---|---|---|---|",
        ]
        for name, spec in TASKS.items():
            if spec.kind == CLASSIFICATION:
                mark = "✅" if row[f"hit_{name}"] else "❌"
                lines.append(f"| {name} | {row[f'pred_{name}']} ({row[f'prob_{name}']:.0%}) | {row[f'true_{name}']} | {mark} |")
            else:
                off = row[f"abs_error_{name}"] / row[f"true_{name}"]
                lines.append(f"| {name} | ${row[f'pred_{name}']:,.0f} | ${row[f'true_{name}']:,.0f} | off by {off:.0%} |")
        return "\n".join(lines)

    last = history[-1]
    minutes = last["seconds"] / 60
    md = [
        f"# HR-Jev results · `{args.name}`",
        "",
        f"Checkpoint `{ckpt}`: lowest validation loss at epoch {best_epoch} of {len(history)}, "
        + f"{minutes:.0f} min of training{' on ' + args.hardware if args.hardware else ''}.",
        "Generated by `python scripts/make_report.py`; every number below is reproducible with",
        f"`hr-evaluate --checkpoint {ckpt}`.",
        "",
        "## Training curves",
        "",
        f"![training curves](../images/{args.name}_training_curves.png)",
        "",
        "| epoch | train loss | val loss | Attrition bal. acc. | Attrition AUC | JobLevel acc. | JobSatisfaction bal. acc. | Income MAE | Income R² |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for h in history:
        v = h["val"]
        md.append(
            f"| {h['epoch']}{' ★' if h['epoch'] == best_epoch else ''} | {h['train_loss']:.3f} | {h['val_loss']:.3f} | "
            f"{v['Attrition']['balanced_accuracy']:.1%} | {v['Attrition']['roc_auc']:.3f} | "
            f"{v['JobLevel']['accuracy']:.1%} | {v['JobSatisfaction']['balanced_accuracy']:.1%} | "
            f"${v['MonthlyIncome']['mae']:,.0f} | {v['MonthlyIncome']['r2']:.3f} |"
        )
    md += [
        "",
        "★ = epoch saved as `best.pt` (lowest validation loss). An income R² below zero means "
        + "the head is still worse than predicting the mean income (R² = 1 − model error / mean error).",
        "",
        "## Test split: HR-Jev vs the baselines",
        "",
    ]
    md.append(to_markdown(results))
    md += ["## Confusion matrices (test split)", ""]
    for name, spec in TASKS.items():
        if spec.kind == CLASSIFICATION:
            md += [f"**{name}**", "", confusion_markdown(spec, truth_idx[name], pred_idx[name]), ""]
    md += [
        "## Monthly income: predicted vs actual",
        "",
        f"![income scatter](../images/{args.name}_income.png)",
        "",
        "## Worked examples",
        "",
        "Every row of the test split, with the sentence the model read, its four predictions and",
        f"the truth, is in [`{args.name}_test_predictions.csv`]({args.name}_test_predictions.csv).",
        "A few of them, chosen to show what works and what does not:",
        "",
        "### Where everything is right",
        "",
    ]
    for _, row in good.iterrows():
        md += [example(row), ""]
    md += ["### Leavers the model caught", ""]
    for _, row in leaver_hits.iterrows():
        md += [example(row), ""]
    md += ["### Attrition mistakes (the most confident ones)", ""]
    for _, row in att_miss.iterrows():
        md += [example(row), ""]
    md += ["### Largest income errors", ""]
    for _, row in inc_miss.iterrows():
        md += [example(row), ""]
    (examples / f"{args.name}_report.md").write_text("\n".join(md))
    print("\n".join(md[: md.index("## Confusion matrices (test split)")]))
    print(f"\nwrote {examples / f'{args.name}_report.md'}")


if __name__ == "__main__":
    main()
