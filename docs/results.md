# Results

**Measured HR-Jev results** (`hr-train --init distilgpt2`, 8 epochs, 44 min on an 8-core
laptop CPU) are in [examples/distilgpt2_report.md](examples/distilgpt2_report.md): training
curves, the test tables below with the model's column filled in, confusion matrices, an
income scatter plot and worked examples. Every test employee's inputs, truth and predictions
are in [examples/distilgpt2_test_predictions.csv](examples/distilgpt2_test_predictions.csv).

Summary on the 147 test employees:

| Task | Metric | HR-Jev (DistilGPT-2) | linear | trivial |
|---|---|---:|---:|---:|
| Attrition | ROC-AUC | 0.737 | 0.853 | 0.500 |
| Attrition | balanced accuracy | 65.7% | 75.3% | 50.0% |
| JobLevel | accuracy | 66.0% | 76.2% | 33.3% |
| JobSatisfaction | balanced accuracy | 22.1% | 21.3% | 25.0% |
| MonthlyIncome | MAE | $1,491 | $1,388 | $3,672 |
| MonthlyIncome | R² | 0.845 | 0.825 | −0.191 |

The model beats the trivial baseline everywhere it can be beaten, beats the linear model on
income (R², RMSE) and is behind it on attrition and job level. Job satisfaction is at
chance for every model.

To reproduce or to add a run (e.g. full GPT-2 on a Kaggle/Colab GPU with
[the notebook](../notebooks/train_colab_kaggle.ipynb)):

```bash
hr-train --init distilgpt2 --output-dir checkpoints/distilgpt2       # or the notebook
hr-evaluate --checkpoint checkpoints/distilgpt2/best.pt              # tables vs baselines
python scripts/make_report.py --checkpoint checkpoints/distilgpt2/best.pt --name distilgpt2
```

`make_report.py` writes the curves and the scatter plot to `docs/images/`, and the report
and the per-row CSV to `docs/examples/`. Checkpoints (337 MB for DistilGPT-2) are not
committed; the run is seeded, so the command above gives the same numbers on the same
machine, and numbers within a point or two elsewhere (147 test rows is a small sample).

## Baselines to beat (measured)

Same test split (147 employees never used for training or model selection), same 27
input columns as HR-Jev. Reproduce with `hr-evaluate` (it computes them every time).

- **Linear:** logistic regression (class-balanced for attrition) / ridge regression on
  log(income), with one-hot categories and standardised numbers.
- **Trivial:** always the most frequent class in the training split / the median income.

**Attrition** (24 of the 147 test employees leave)

| metric | linear | trivial |
|---|---:|---:|
| accuracy | 75.5% | 83.7% |
| balanced accuracy | 75.3% | 50.0% |
| macro-F1 | 0.669 | 0.456 |
| ROC-AUC | 0.853 | 0.500 |

**JobLevel**

| metric | linear | trivial |
|---|---:|---:|
| accuracy | 76.2% | 33.3% |
| balanced accuracy | 63.8% | 20.0% |
| macro-F1 | 0.639 | 0.100 |

**JobSatisfaction**

| metric | linear | trivial |
|---|---:|---:|
| accuracy | 24.5% | 36.1% |
| balanced accuracy | 21.3% | 25.0% |
| macro-F1 | 0.190 | 0.133 |

**MonthlyIncome**

| metric | linear | trivial |
|---|---:|---:|
| MAE | $1,388 | $3,672 |
| RMSE | $2,076 | $5,418 |
| R² | 0.825 | −0.191 |

## How to read these numbers

- **Attrition: look at ROC-AUC and balanced accuracy, not accuracy.** 84% of
  employees stay, so "always stays" is 83.7% accurate while finding no leaver at all.
  The class-balanced linear model is *less* accurate (75.5%) but actually finds leavers
  (balanced accuracy 75.3% vs 50%).
  The first version of this repo reported 80.9% accuracy (on its own random split, where
  ~84% also stay), which is below doing nothing.
- **JobSatisfaction is noise in this dataset.** The linear model is at chance
  (21% balanced accuracy vs 25% for guessing). Expect HR-Jev to be there too; a much
  higher number on 147 rows would more likely be luck than skill.
- **JobLevel and MonthlyIncome are easy for a linear model** (76% accuracy, R² 0.83):
  they follow experience and job role closely, so HR-Jev needs to match that first.
- **A negative R² is not a bug.** R² = 1 − (model error) / (error of always predicting the
  mean). A model worse than the mean gets a negative value; the trivial baseline (the median
  income) scores −0.19, and so does HR-Jev in its first epoch before the income head has learned.
- **147 test rows is small.** For attrition that is 24 leavers: one more or one fewer
  correct moves the leaver recall by about 4 points. Differences of a point or two
  between models are not meaningful; repeating training with several seeds is the next
  step (see the README's next steps).

## Is a GPT worth it here?

Honestly, for 1,470 rows of clean tabular data, a linear model or gradient boosting is
the stronger, cheaper default, and HR-Jev may not beat the linear baseline. What the
project shows is the *technique* (serialising rows to text and fine-tuning a
pretrained Transformer with several heads), which pays off when columns are many,
messy or free-text, or when there is little labelled data for a new task. The
baselines are here so that any claim about HR-Jev can be checked.
