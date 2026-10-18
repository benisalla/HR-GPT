# Training guide

## Install

```bash
python -m venv .venv
source .venv/bin/activate              # Windows: .venv\Scripts\activate
pip install -e '.[app,dev]'
```

PyTorch: the command above installs the default build for your platform. For a
specific CUDA version, or a smaller CPU-only build, install torch first with the
selector at [pytorch.org](https://pytorch.org/get-started/locally/), e.g.

```bash
pip install torch --index-url https://download.pytorch.org/whl/cpu    # CPU only
pip install -e '.[app,dev]'
```

## Train

```bash
hr-train                                   # GPT-2 small, all defaults (GPU recommended)
hr-train --init distilgpt2                 # half the layers, ~2x faster (good on CPU)
```

The first run downloads the dataset (230 KB) to `data/` and the GPT-2 weights
(~500 MB, cached by Hugging Face). After each epoch it prints the validation loss and
the main metric of each task, and saves `best.pt` when the validation loss improves.

Output in `--output-dir` (default `checkpoints/run/`):

| File | What |
|---|---|
| `best.pt` | weights + everything needed to use them (configs, tasks, income scaling) at the lowest validation loss |
| `last.pt` | the same after the last epoch |
| `tokenizer/` | the GPT-2 tokenizer with `<|pad|>` (inference works offline) |
| `history.json` | train/val loss and every validation metric, per epoch |

Then measure it once on the test split and try it:

```bash
hr-evaluate --checkpoint checkpoints/run/best.pt
hr-app      --checkpoint checkpoints/run/best.pt
```

## On a free GPU: Kaggle or Colab

[`notebooks/train_colab_kaggle.ipynb`](../notebooks/train_colab_kaggle.ipynb) is
self-contained (all the code inside, same defaults). Upload it, turn on the GPU
(and Internet on Kaggle), **Run all**. It shows a live dashboard, the test results
next to the baselines, a few example predictions, and gives you a zip with
`best.pt` to use with `hr-app` on your computer.

The notebook is generated from the package, so it never drifts from the code:

```bash
python notebooks/build_notebook.py
```

## How long

The dataset is small: 1,176 training employees with ~165 tokens each, so one epoch
is 74 steps of 16 employees. **Use a GPU** (the notebook on Kaggle or Colab). On a
CPU, prefer `--init distilgpt2`; on a laptop CPU shared with other jobs, even that
took over 30 minutes for one epoch. If you run out of memory, lower `--batch-size`.

## All flags

| Flag | Default | Meaning |
|---|---|---|
| `--init` | `gpt2` | `gpt2`, `distilgpt2` or `scratch` (random weights) |
| `--layers --heads --embd` | 12 / 12 / 768 | model size, only with `--init scratch` |
| `--dropout` | 0.1 | dropout in the backbone |
| `--epochs` | 8 | maximum number of epochs |
| `--batch-size` | 16 | employees per step |
| `--lr` | 3e-5 | backbone learning rate (pretrained: keep it small) |
| `--head-lr` | 1e-3 | task heads learning rate (new layers) |
| `--weight-decay` | 0.01 | AdamW weight decay, matrices only |
| `--patience` | 3 | stop after this many epochs without a better val loss (0 = never) |
| `--train-embeddings` | off | also fine-tune the 38.6M-parameter token embedding table |
| `--seed` | 1337 | random seed (the data split has its own fixed seed) |
| `--device` | `auto` | `auto` (cuda > mps > cpu), `cpu`, `cuda`, `mps` |
| `--precision` | `auto` | `auto` (bf16 on GPUs that support it), `fp32`, `bf16` |
| `--limit` | none | train on the first N employees only (quick checks) |
| `--data-dir` | `data` | where the CSV and the splits go |
| `--output-dir` | `checkpoints/run` | where checkpoints go |

Task settings (classes, loss weights, class balancing, log target) are in
`TASKS` in `src/hr_jev/config.py`.

## Things to try

- `--init scratch`: same architecture, random weights. How much does pretraining help on 1.2k rows?
- `--train-embeddings`: does fine-tuning the embedding table help or overfit?
- `loss_weight` in `TASKS`: give attrition more weight if it is the task you care about.
- Drop the noise columns (`DailyRate`, `HourlyRate`, `MonthlyRate`: random numbers in
  this dataset) from `FEATURES` for shorter sentences.
