"""Build notebooks/train_colab_kaggle.ipynb: a self-contained training notebook.

The code cells are copied from src/hr_jev/*.py at build time (module docstrings become
the markdown above each cell), so the notebook is identical to the package and needs no
GitHub access. Regenerate after changing src/hr_jev:

    python notebooks/build_notebook.py
"""

import ast
from pathlib import Path

import nbformat as nbf

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src" / "hr_jev"
OUT = ROOT / "notebooks" / "train_colab_kaggle.ipynb"

SKIP = {"main", "build_parser"}  # command-line entry points are not needed in a notebook
cells: list = []


def md(text: str) -> None:
    cells.append(nbf.v4.new_markdown_cell(text.strip()))


def code(text: str) -> None:
    cells.append(nbf.v4.new_code_cell(text.strip()))


def module(name: str, skip: set[str] = SKIP) -> tuple[str, str]:
    """(docstring, body) of src/hr_jev/<name>.py without imports, CLI code and __main__."""
    text = (SRC / f"{name}.py").read_text()
    lines = text.splitlines()
    tree = ast.parse(text)
    doc = ast.get_docstring(tree) or ""
    parts = []
    for node in tree.body:
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            continue
        if isinstance(node, ast.Expr) and isinstance(getattr(node, "value", None), ast.Constant):
            continue  # the module docstring
        if isinstance(node, ast.If):  # if __name__ == "__main__"
            continue
        if isinstance(node, (ast.FunctionDef, ast.ClassDef)) and node.name in skip:
            continue
        start = min([node.lineno] + [d.lineno for d in getattr(node, "decorator_list", [])])
        parts.append("\n".join(lines[start - 1 : node.end_lineno]))
    return doc, "\n\n\n".join(parts)


def module_cells(title: str, name: str, intro: str = "") -> None:
    doc, body = module(name)
    md(f"{title}\n\n{intro}\n\n*Same code as `src/hr_jev/{name}.py`.*\n\n" + _quote(doc))
    code(body)


def _quote(doc: str) -> str:
    """Docstrings often hold small diagrams: keep them as a code block."""
    first, _, rest = doc.partition("\n")
    return first + ("\n\n```text\n" + rest.strip("\n") + "\n```" if rest.strip() else "")


# ---------------------------------------------------------------------------
md("""
# HR-Jev — training notebook (Kaggle & Colab)

Fine-tune **GPT-2** to read an employee description and predict four things at once:
**attrition** (will they leave?), **job level**, **job satisfaction** and **monthly income**.

The notebook is **self-contained**: all the code is inside, identical to the
[HR-Jev repo](https://github.com/benisalla/HR-Jev), with the same defaults. Upload it and
press **Run all**.

| | Kaggle | Colab |
|---|---|---|
| GPU | *Settings → Accelerator → GPU T4* | *Runtime → Change runtime type → T4 GPU* |
| Internet | *Settings → Internet → On* (data + GPT-2 weights) | on by default |
| Outputs kept in | `/kaggle/working/hr-jev` (saved with "Save Version") | Google Drive if `USE_GOOGLE_DRIVE = True` |

It also runs on a CPU, but slowly: there, set `INIT = "distilgpt2"`.

**Before the long run:** set `SMOKE_TEST = True` in section 2.2 and *Run all* once
(about a minute): it checks the whole notebook with a tiny model. Then set it back to
`False` and run again.

**Sections:** 1. Setup · 2. Config · 3. Data · 4. Model (how the architecture works) ·
5. Training (live dashboard) · 6. Test results vs baselines · 7. Try the model · 8. Download
""")

md("""
## 1. Setup

Kaggle and Colab already have everything we need (PyTorch, transformers, pandas,
scikit-learn, matplotlib). Nothing to install.
""")
code("""
import hashlib
import json
import math
import random
import shutil
import time
import urllib.request
from collections.abc import Callable, Sequence
from contextlib import nullcontext
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from IPython.display import Markdown, clear_output, display
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    mean_absolute_error,
    r2_score,
    roc_auc_score,
)
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from torch import Tensor, nn
from torch.nn import functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader, Dataset

gpu = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "none"
print("torch", torch.__version__, "| GPU:", gpu)
""")

# ---------------------------------------------------------------------------
module_cells("## 2. Config\n\n### 2.1 Config classes", "config")

md("""
### 2.2 Settings for this run

The defaults are the repo's `hr-train` defaults. `SMOKE_TEST = True` runs the whole
notebook in about a minute with a tiny random model (to check that everything works).
""")
code("""
SMOKE_TEST = False  # True = tiny quick check of the whole notebook
USE_GOOGLE_DRIVE = False  # Colab only: keep outputs on Google Drive
INIT = "gpt2"  # "gpt2" (12 layers) | "distilgpt2" (6 layers, 2x faster) | "scratch"
EPOCHS = 8

if Path("/kaggle/working").exists():
    BASE = Path("/kaggle/working/hr-jev")
elif USE_GOOGLE_DRIVE:
    from google.colab import drive

    drive.mount("/content/drive")
    BASE = Path("/content/drive/MyDrive/hr-jev")
else:
    BASE = Path("hr-jev")

data_config = DataConfig(data_dir=BASE / "data")
model_config = ModelConfig(init=INIT)
train_config = TrainConfig(output_dir=BASE / "run", epochs=EPOCHS)
if SMOKE_TEST:
    model_config = ModelConfig(init="scratch", n_layer=2, n_head=2, n_embd=64)
    train_config = TrainConfig(output_dir=BASE / "smoke", epochs=2, limit=96, lr=1e-3, log_every=2)
print(model_config, train_config, sep="\\n")
""")

# ---------------------------------------------------------------------------
module_cells(
    "## 3. Data\n\n### 3.1 The employee record",
    "features",
    "Each employee has 27 input columns. They are written as one English sentence, which is "
    "what GPT-2 is good at reading. The four columns we predict are never in the sentence.",
)
module_cells("### 3.2 Download, split, tokenize, batch", "data")
md("""
### 3.3 Look at the data

[IBM HR Analytics Employee Attrition](https://www.kaggle.com/datasets/pavansubhasht/ibm-hr-analytics-attrition-dataset):
1,470 fictional employees created by IBM data scientists. The split is 80/10/10,
stratified on attrition (16% of employees leave).
""")
code("""
paths = prepare_data(data_config)
train_frame = load_split(data_config, "train")
example = frame_to_records(train_frame)[0]
print(record_to_text(example), "\\n")
print({c: example[c] for c in TARGET_COLUMNS})

fig, axes = plt.subplots(1, 4, figsize=(16, 3))
for ax, col in zip(axes, TARGET_COLUMNS):
    if col == "MonthlyIncome":
        ax.hist(train_frame[col], bins=30)
    else:
        train_frame[col].value_counts().sort_index().plot.bar(ax=ax)
    ax.set_title(col)
plt.tight_layout()
plt.show()
""")

# ---------------------------------------------------------------------------
md("""
## 4. Model

### 4.0 How the architecture works

```text
employee row (27 columns; the 4 targets are never inputs)
   │  record_to_text (section 3.1)
   ▼
"The employee is 41 years old, is female, ..., works overtime, ..."
   │  GPT-2 tokenizer + <eos>, right-padded with <|pad|>
   ▼
input_ids [B, T]   (T ≈ 165)          lengths [B]
   │  token embedding + position embedding
   ▼
[B, T, 768]
   │  12 × Block:  x = x + Attention(LayerNorm(x))   (causal: each token sees only the past)
   │               x = x + MLP(LayerNorm(x))          (768 → 3072 → 768)
   │  final LayerNorm
   ▼
[B, T, 768]
   │  take each row's last real token (<eos>, position lengths−1):
   │  it is the only token that has read the whole sentence, and padding comes after it
   ▼
[B, 768] ─┬─► Attrition head        [B, 2] → softmax → Stays / Leaves
          ├─► JobLevel head         [B, 5] → softmax → Level 1..5
          ├─► JobSatisfaction head  [B, 4] → softmax → Low..Very High
          └─► MonthlyIncome head    [B]    → exp(z·std + mean) → dollars
```

- **Why a sentence?** GPT-2 already knows what "works overtime" or "Research Director"
  mean. With only 1,176 training employees, that prior knowledge is the point.
- **Backbone:** GPT-2 written from scratch below, loaded with OpenAI's GPT-2 weights
  (the attention/MLP matrices are transposed: Hugging Face stores them as `Conv1D`).
  The token embedding table (38.6M params) is frozen; the rest is fine-tuned.
- **Heads:** `Linear(768, 768) → GELU → Dropout → Linear(768, k)`, one per task. They
  all read the same vector, so **one forward pass answers all four questions** and the
  four tasks train the backbone together.
- **Loss:** sum of cross-entropy (×3; attrition is class-weighted because only 16%
  leave) and MSE on standardised log(income).
- **Optimiser:** AdamW, 3e-5 for the pretrained backbone and 1e-3 for the new heads,
  warmup + cosine, gradient clipping, early stopping on the validation loss.
""")
module_cells("### 4.1 The code", "model")
md("""
### 4.2 Check the shapes on one real batch
Builds the model and pushes 4 real employees through it, printing each step's shape.
""")
code("""
tokenizer = build_tokenizer("gpt2")
demo_targets = TargetEncoder.fit(TASKS, train_frame)
demo_set = HRDataset(train_frame.iloc[:4], tokenizer, demo_targets, data_config.max_tokens)
batch = Collator(tokenizer.pad_token_id)([demo_set[i] for i in range(4)])
demo_model = build_model(replace(model_config, vocab_size=len(tokenizer)), TASKS).eval()
with torch.no_grad():
    pooled = demo_model.encode(batch["input_ids"], batch["lengths"])
    outputs = demo_model(batch["input_ids"], batch["lengths"])
print("input_ids        ", tuple(batch["input_ids"].shape), "  lengths", batch["lengths"].tolist())
print("pooled <eos> state", tuple(pooled.shape))
for name, out in outputs.items():
    print(f"{name:17s}", tuple(out.shape))
print(f"parameters: {demo_model.num_parameters() / 1e6:.1f}M")
del demo_model
""")
module_cells(
    "### 4.3 Metrics and inference helpers",
    "metrics",
    "Accuracy alone is misleading with 84% stayers, so we also look at balanced accuracy, "
    "macro-F1, ROC-AUC, MAE and R².",
)
module_cells("### 4.4 Predictor", "predict")

# ---------------------------------------------------------------------------
module_cells("## 5. Training", "train")
md("""
### 5.1 Train with a live dashboard

**What to expect.** Measured on the same test split, the bar to clear is the linear
baseline: attrition ROC-AUC 0.853, job level accuracy 76.2%, income MAE $1,388
(R² 0.825). Job satisfaction is at chance for every model (balanced accuracy ≈ 25%):
it is not predictable from these columns.

The plots refresh after each epoch: the losses on the left, then the main validation
metric of each task (balanced accuracy for the classifiers, R² for income).
Dashed lines are the "always guess the majority" level.
""")
code("""
CHANCE = {"Attrition": 0.5, "JobLevel": 0.2, "JobSatisfaction": 0.25, "MonthlyIncome": 0.0}


def dashboard(history):
    clear_output(wait=True)
    epochs = [h["epoch"] for h in history]
    _, axes = plt.subplots(1, 5, figsize=(20, 3.2))
    axes[0].plot(epochs, [h["train_loss"] for h in history], "o-", label="train")
    axes[0].plot(epochs, [h["val_loss"] for h in history], "o-", label="val")
    axes[0].set_title("loss")
    axes[0].legend()
    for ax, name in zip(axes[1:], TASKS):
        ax.plot(epochs, [h["val_headline"][name] for h in history], "o-")
        ax.axhline(CHANCE[name], ls="--", c="gray")
        ax.set_title(f"{name} ({HEADLINE[TASKS[name].kind]})")
    for ax in axes:
        ax.set_xlabel("epoch")
    plt.tight_layout()
    plt.show()
    last = history[-1]
    print(f"epoch {last['epoch']}: val_loss={last['val_loss']:.4f}  ({last['seconds']:.0f}s)")
    print(format_metrics(TASKS, last["val"]))


summary = train(data_config, model_config, train_config, on_epoch_end=dashboard)
print(f"best epoch: {summary['best_epoch']} (val_loss={summary['best_val_loss']:.4f})")
""")

# ---------------------------------------------------------------------------
module_cells(
    "## 6. Test results vs baselines",
    "evaluate",
    "The test split (147 employees) is used once, here, with the best checkpoint.",
)
code("""
results = evaluate_checkpoint(train_config.output_dir / BEST_CHECKPOINT, data_config)
(train_config.output_dir / "test_results.json").write_text(json.dumps(results, indent=2))
(train_config.output_dir / "test_results.md").write_text(to_markdown(results))
display(Markdown(to_markdown(results)))
""")

# ---------------------------------------------------------------------------
md("""
## 7. Try the model

Load `best.pt` the way the repo's `hr-predict` and `hr-app` do, predict a few test
employees and compare with the truth. Then change one field and see what moves.
""")
code("""
predictor = Predictor.from_checkpoint(train_config.output_dir / BEST_CHECKPOINT)
test_rows = frame_to_records(load_split(data_config, "test"))
for row in test_rows[:3]:
    print(format_prediction(predictor.predict(row), actual_values(row, predictor.tasks)), "\\n")

what_if = {**test_rows[0], "OverTime": "Yes", "YearsSinceLastPromotion": 8}
print("Same employee, with overtime and no promotion for 8 years:")
print(format_prediction(predictor.predict(what_if)))
""")

# ---------------------------------------------------------------------------
md("""
## 8. Download the trained model

Zips `best.pt`, the tokenizer and the results. On your computer, with the repo installed:

```bash
unzip hr-jev-model.zip -d checkpoints/run
hr-app --checkpoint checkpoints/run/best.pt
```
""")
code("""
bundle = BASE / "bundle"
shutil.rmtree(bundle, ignore_errors=True)
bundle.mkdir(parents=True)
for name in [BEST_CHECKPOINT, "history.json", "test_results.json", "test_results.md"]:
    shutil.copy(train_config.output_dir / name, bundle / name)
shutil.copytree(train_config.output_dir / TOKENIZER_DIR, bundle / TOKENIZER_DIR)
archive = shutil.make_archive(str(BASE / "hr-jev-model"), "zip", bundle)
print("saved", archive, f"({Path(archive).stat().st_size / 1e6:.0f} MB)")
try:
    from google.colab import files

    files.download(archive)
except ImportError:
    print("Kaggle: find it under Output -> hr-jev/hr-jev-model.zip")
""")

nb = nbf.v4.new_notebook()
nb.cells = cells
for i, cell in enumerate(cells):
    cell.id = f"cell-{i:02d}"  # stable ids: rebuilding does not change the file
nb.metadata = {
    "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
    "language_info": {"name": "python"},
    "accelerator": "GPU",
}
nbf.write(nb, OUT)
print(f"wrote {OUT} ({len(cells)} cells)")
