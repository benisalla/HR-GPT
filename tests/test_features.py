"""The schema: every record becomes one well-formed sentence, the same in training and in the app."""

import pandas as pd
import pytest

from hr_llm.features import (
    FEATURES,
    FEATURES_BY_NAME,
    TARGET_COLUMNS,
    default_record,
    record_to_text,
    validate_record,
)

from .conftest import synthetic_rows


def test_default_record_is_valid_and_readable():
    text = record_to_text(default_record())
    assert text.startswith("The employee is 35 years old, is male, is married")
    assert text.endswith(".")


def test_every_feature_appears_once_per_sentence():
    text = record_to_text(default_record())
    assert text.count(", ") == len(FEATURES) - 1


@pytest.mark.parametrize(
    ("name", "value", "phrase"),
    [
        # The CSV codes are 1-based. The old code read them as 0-based ("1" became "college").
        ("Education", 1, "has below college education"),
        ("Education", 5, "has doctor education"),
        ("PerformanceRating", 3, "has a performance rating of excellent"),
        ("PerformanceRating", 4, "has a performance rating of outstanding"),
        ("JobInvolvement", 1, "shows low job involvement"),
        ("EnvironmentSatisfaction", 4, "reports very high environment satisfaction"),
        ("WorkLifeBalance", 1, "has bad work-life balance"),
        ("BusinessTravel", "Non-Travel", "does not travel for work"),
        ("OverTime", "Yes", "works overtime"),
        ("OverTime", "No", "does not work overtime"),
        ("NumCompaniesWorked", 1, "has worked at 1 company before"),
        ("YearsAtCompany", 1, "has spent 1 year at the company"),
        ("YearsAtCompany", 0, "has spent 0 years at the company"),
    ],
)
def test_phrases(name, value, phrase):
    assert FEATURES_BY_NAME[name].phrase(value) == phrase


@pytest.mark.parametrize("target", TARGET_COLUMNS)
def test_targets_never_leak_into_the_sentence(target):
    row = synthetic_rows(1)[0]
    text = record_to_text(row)
    assert target not in text
    # MonthlyIncome's value must not appear either
    assert f" {row['MonthlyIncome']} " not in text


def test_extra_columns_are_ignored():
    rec = default_record()
    assert record_to_text({**rec, "Attrition": "Yes", "Foo": 1}) == record_to_text(rec)


def test_numbers_from_the_ui_are_rounded():
    rec = {**default_record(), "Age": 41.0, "Education": 2.0}
    assert validate_record(rec)["Age"] == 41
    assert validate_record(rec)["Education"] == 2


@pytest.mark.parametrize(
    ("change", "message"),
    [
        ({"Gender": "Robot"}, "'Gender' must be one of"),
        ({"Education": 9}, "'Education' must be one of"),
        ({"Age": "old"}, "'Age' must be a number"),
        ({"Age": -1}, "cannot be negative"),
        ({"Age": None}, "missing value for 'Age'"),
    ],
)
def test_invalid_records_are_rejected(change, message):
    with pytest.raises(ValueError, match=message):
        validate_record({**default_record(), **change})


def test_choices_cover_the_real_csv_values():
    """Values copied from the IBM CSV header/value counts (see docs/architecture.md)."""
    real = {
        "BusinessTravel": {"Travel_Rarely", "Travel_Frequently", "Non-Travel"},
        "Department": {"Sales", "Research & Development", "Human Resources"},
        "EducationField": {
            "Life Sciences",
            "Other",
            "Medical",
            "Marketing",
            "Technical Degree",
            "Human Resources",
        },
        "Gender": {"Female", "Male"},
        "JobRole": {
            "Sales Executive",
            "Research Scientist",
            "Laboratory Technician",
            "Manufacturing Director",
            "Healthcare Representative",
            "Manager",
            "Sales Representative",
            "Research Director",
            "Human Resources",
        },
        "MaritalStatus": {"Single", "Married", "Divorced"},
        "OverTime": {"Yes", "No"},
        "Education": {1, 2, 3, 4, 5},
        "PerformanceRating": {3, 4},
        "WorkLifeBalance": {1, 2, 3, 4},
    }
    for name, values in real.items():
        assert values <= set(FEATURES_BY_NAME[name].choices), name


def test_csv_round_trip_keeps_the_same_sentence(tmp_path):
    rows = synthetic_rows(5)
    path = tmp_path / "x.csv"
    pd.DataFrame(rows).to_csv(path, index=False)
    back = pd.read_csv(path).to_dict(orient="records")
    assert [record_to_text(r) for r in back] == [record_to_text(r) for r in rows]
