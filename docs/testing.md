# Testing guide

Three levels of checks:

1. **Unit tests** (`pytest`): every module, CPU only, offline, ~3 minutes.
2. **Command-line smoke test**: the real commands on the real data with a tiny model, ~1 minute.
3. **Web app test**: click through the interface (by hand, or with the Playwright script below).

## 1. Unit tests

```bash
pytest                         # everything
pytest tests/test_model.py     # one file
pytest -k padding -v           # tests whose name contains "padding"
ruff check src tests notebooks # lint
```

No GPU and no internet needed: the tests use a synthetic CSV with the real
columns, a small word-level tokenizer built in memory, and tiny models.

| File | What it proves |
|------|----------------|
| `test_features.py` | each column turns into the right words (the 1-based codes: Education 1 is "below college", PerformanceRating 3 is "excellent"); **targets never appear in the sentence**; bad values from the app give clear errors; the choices cover every value in the real CSV |
| `test_data.py` | preparation drops the ID/constant columns and writes 3 disjoint splits; the split is reproducible and keeps the attrition rate; a wrong MD5 is caught; targets encode and decode (income round-trips to the dollar); balanced weights favour leavers; the collator right-pads and keeps `<eos>` last |
| `test_model.py` | output shapes; **padding cannot change a prediction**; attention is causal; every head gets gradients; optimizer groups cover each parameter once (heads get their own LR); frozen embeddings stay out of the optimizer; a tiny model can overfit a batch; **loading GPT-2 weights gives exactly Hugging Face GPT-2's hidden states** |
| `test_train.py` | warmup + cosine LR; a full training run lowers the loss and writes `best.pt`, `last.pt`, `history.json`, `tokenizer/`; embeddings really stay frozen; `best.pt` reloads and predicts all four tasks deterministically; a missing checkpoint gives a helpful error; evaluation reports the model next to both baselines |
| `test_app.py` | the results table (prediction, confidence, true value, ✅/❌, "off by x%"); the app builds from a real checkpoint |

## 2. Command-line smoke test (real data, tiny model)

Checks that the real commands work together, end to end:

```bash
hr-train --init scratch --layers 2 --heads 4 --embd 64 --limit 64 --epochs 2 \
         --lr 1e-3 --output-dir checkpoints/smoke
hr-predict  --checkpoint checkpoints/smoke/best.pt --test-row 0
hr-evaluate --checkpoint checkpoints/smoke/best.pt
```

The numbers will be bad (64 rows, 2 epochs, random weights). This only checks the plumbing.

## 3. Web app test

```bash
hr-app --checkpoint checkpoints/run/best.pt     # http://127.0.0.1:7860
```

By hand, in two minutes:

| Do | Expect |
|---|---|
| Click **Predict** | 4 rows: attrition, job level, job satisfaction, income |
| Click **🎲 Random test employee** | the form fills in, predictions appear right away with an **Actual** column and ✅ / ❌ |
| Change any field | the status says "Edited by hand"; the next prediction has no Actual column |
| Open **What the model reads** | the sentence shows your edit |
| Empty a number field, then **Predict** | "⚠️ Cannot predict: missing value for …", nothing else changes |
| Click **↺ Reset** | defaults come back |

The same clicks are scripted in [`scripts/capture_demo.py`](../scripts/capture_demo.py),
which also takes the screenshots and the GIF shown in the README:

```bash
pip install playwright && playwright install chromium ffmpeg   # once
hr-app --checkpoint checkpoints/run/best.pt                    # terminal 1
python scripts/capture_demo.py --out docs/images               # terminal 2
```

It loads the page, clicks **Random test employee**, opens both panels, switches
**Works overtime** to Yes and predicts again, and reports any JavaScript error.
