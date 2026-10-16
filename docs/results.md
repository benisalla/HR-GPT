# Results

**No trained HR-GPT checkpoint is published yet.** Train it with
[the notebook](../notebooks/train_colab_kaggle.ipynb) on a free Kaggle or Colab GPU;
its section 6 prints HR-GPT's test results next to the baselines below, and saves
them as `test_results.md`. Paste that table here.

## Baselines to beat (measured)

Same test split (147 employees never used for training or model selection), same 27
input columns as HR-GPT. Reproduce with `hr-evaluate` (it computes them every time).

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
  (21% balanced accuracy vs 25% for guessing). Expect HR-GPT to be there too; a much
  higher number on 147 rows would more likely be luck than skill.
- **JobLevel and MonthlyIncome are easy for a linear model** (76% accuracy, R² 0.83):
  they follow experience and job role closely, so HR-GPT needs to match that first.
- **147 test rows is small.** For attrition that is 24 leavers: one more or one fewer
  correct moves the leaver recall by about 4 points. Differences of a point or two
  between models are not meaningful; repeating training with several seeds is the next
  step (see the README's next steps).

## Is a GPT worth it here?

Honestly, for 1,470 rows of clean tabular data, a linear model or gradient boosting is
the stronger, cheaper default, and HR-GPT may not beat the linear baseline. What the
project shows is the *technique* (serialising rows to text and fine-tuning a
pretrained Transformer with several heads), which pays off when columns are many,
messy or free-text, or when there is little labelled data for a new task. The
baselines are here so that any claim about HR-GPT can be checked.
