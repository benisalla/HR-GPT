# Using a trained model

Every command takes `--checkpoint` (default `checkpoints/run/best.pt`). The tokenizer
is saved next to the checkpoint, so inference works offline.

## Web app

```bash
hr-app --checkpoint checkpoints/run/best.pt            # http://127.0.0.1:7860
hr-app --port 8000 --host 0.0.0.0                      # reachable from your network
hr-app --share                                         # temporary public gradio.live link
```

1. **🎲 Random test employee** loads one of the 147 test employees (never seen in
   training) and predicts right away, with the true answers next to the predictions.
2. Or edit the form (4 tabs: Personal, Job, Pay, Survey) and click **Predict**.
3. **Probabilities** shows every class; **What the model reads** shows the exact sentence.

Screenshots and a GIF of the app, made with `scripts/capture_demo.py`, are in the
README's Demo section.

## Command line

```bash
# a test employee: prints predictions and the true values
hr-predict --test-row 12 --show-text

# your own employee: a JSON file with the 27 input columns
hr-predict --json employee.json
```

It prints one line per task: the predicted class and its probability (or the
income in dollars), plus the true value for test rows. For example, with the
DistilGPT-2 checkpoint from `docs/results.md`:

```
$ hr-predict --checkpoint checkpoints/distilgpt2/best.pt --test-row 4
Attrition        Leaves (76%)   | actual: Leaves
JobLevel         Level 1 (100%)   | actual: Level 1
JobSatisfaction  Very High (32%)   | actual: Low
MonthlyIncome    $2,145   | actual: $2,033
```

All 147 test rows, with predictions, are in
[examples/distilgpt2_test_predictions.csv](examples/distilgpt2_test_predictions.csv).

`employee.json` uses the CSV's own values (e.g. `"Education": 3` for Bachelor):

```json
{
  "Age": 35, "Gender": "Male", "MaritalStatus": "Married", "Education": 3,
  "EducationField": "Life Sciences", "DistanceFromHome": 7,
  "Department": "Research & Development", "JobRole": "Research Scientist",
  "BusinessTravel": "Travel_Rarely", "OverTime": "No", "NumCompaniesWorked": 2,
  "TotalWorkingYears": 10, "YearsAtCompany": 5, "YearsInCurrentRole": 3,
  "YearsSinceLastPromotion": 1, "YearsWithCurrManager": 3, "TrainingTimesLastYear": 3,
  "DailyRate": 800, "HourlyRate": 65, "MonthlyRate": 14000, "PercentSalaryHike": 14,
  "StockOptionLevel": 1, "PerformanceRating": 3, "JobInvolvement": 3,
  "EnvironmentSatisfaction": 3, "RelationshipSatisfaction": 3, "WorkLifeBalance": 3
}
```

The allowed values for each column are in `src/hr_jev/features.py`. A wrong value
gives a clear error, e.g. `'Gender' must be one of: Male, Female (got 'M')`.

## Python

```python
from hr_jev.features import default_record
from hr_jev.predict import Predictor

predictor = Predictor.from_checkpoint("checkpoints/run/best.pt")
employee = {**default_record(), "OverTime": "Yes", "YearsSinceLastPromotion": 8}
predictor.predict(employee)
# {'Attrition': {'label': 'Stays' or 'Leaves', 'probabilities': {'Stays': p, 'Leaves': 1 - p}},
#  'JobLevel': {'label': 'Level 1'..'Level 5', 'probabilities': {...}},
#  'JobSatisfaction': {'label': ..., 'probabilities': {...}},
#  'MonthlyIncome': {'value': dollars_per_month}}

predictor.predict_many(list_of_employees)   # batched: {task: probs [N, C] or values [N]}
```

## Evaluate on the test split

```bash
hr-evaluate --checkpoint checkpoints/run/best.pt
```

Prints a table of the model next to the two baselines and writes `test_results.json`
and `test_results.md` next to the checkpoint. See [results.md](results.md).

## A word of caution

The data is fictional (made by IBM to teach analytics), small, and contains
gender, age and marital status. The model is a learning project: **do not use it to
make decisions about real people.**
