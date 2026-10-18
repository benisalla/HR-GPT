"""The employee record: which columns exist, how they look in the UI, how they read as text.

This file is the single source of truth for the data schema. Training, evaluation,
the command-line predictor and the web app all use it, so a record is always turned
into exactly the same sentence everywhere.

Values are kept as they appear in the IBM CSV (e.g. Education = 1..5,
BusinessTravel = "Travel_Rarely"); only the sentence uses human words.
"""

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

# Ordinal scales used by several survey columns (the CSV codes start at 1)
LEVEL4 = {1: "Low", 2: "Medium", 3: "High", 4: "Very High"}
EDUCATION = {1: "Below College", 2: "College", 3: "Bachelor", 4: "Master", 5: "Doctor"}
PERFORMANCE = {1: "Low", 2: "Good", 3: "Excellent", 4: "Outstanding"}
WORK_LIFE = {1: "Bad", 2: "Good", 3: "Better", 4: "Best"}


@dataclass(frozen=True)
class Feature:
    """One input column."""

    name: str  # column name in the CSV
    label: str  # label in the web app
    group: str  # tab in the web app
    phrase: Callable[[Any], str]  # value -> words, e.g. 41 -> "is 41 years old"
    default: Any  # value shown in the web app before anything is loaded
    choices: dict[Any, str] = field(default_factory=dict)  # categorical: CSV value -> label
    minimum: float | None = None  # numeric: allowed range in the web app
    maximum: float | None = None

    @property
    def is_categorical(self) -> bool:
        return bool(self.choices)


def _plural(n: int, word: str) -> str:
    return f"{n} {word}" if n == 1 else f"{n} {word}s"


def _travel(value: str) -> str:
    return {
        "Non-Travel": "does not travel for work",
        "Travel_Rarely": "travels rarely for work",
        "Travel_Frequently": "travels frequently for work",
    }[value]


def _num(name: str, label: str, group: str, template: str, default: int, lo: int, hi: int):
    return Feature(
        name, label, group, lambda v: template.format(int(v)), default, minimum=lo, maximum=hi
    )


def _years(name: str, label: str, group: str, template: str, default: int, hi: int):
    return Feature(
        name,
        label,
        group,
        lambda v: template.format(_plural(int(v), "year")),
        default,
        minimum=0,
        maximum=hi,
    )


def _scale(name: str, label: str, group: str, template: str, scale: dict, default: int):
    return Feature(
        name,
        label,
        group,
        lambda v: template.format(scale[int(v)].lower()),
        default,
        choices=dict(scale),
    )


def _cat(name: str, label: str, group: str, template: str, values: list[str], default: str):
    return Feature(
        name,
        label,
        group,
        lambda v: template.format(v.lower()),
        default,
        choices={v: v for v in values},
    )


PERSONAL, JOB, PAY, SURVEY = "Personal", "Job", "Pay", "Survey"

# The order here is the order of the phrases in the sentence.
FEATURES: list[Feature] = [
    _num("Age", "Age", PERSONAL, "is {} years old", 35, 18, 60),
    Feature(
        "Gender",
        "Gender",
        PERSONAL,
        lambda v: f"is {v.lower()}",
        "Male",
        choices={"Male": "Male", "Female": "Female"},
    ),
    _cat(
        "MaritalStatus",
        "Marital status",
        PERSONAL,
        "is {}",
        ["Single", "Married", "Divorced"],
        "Married",
    ),
    _scale("Education", "Education", PERSONAL, "has {} education", EDUCATION, 3),
    _cat(
        "EducationField",
        "Education field",
        PERSONAL,
        "studied {}",
        ["Life Sciences", "Medical", "Marketing", "Technical Degree", "Human Resources", "Other"],
        "Life Sciences",
    ),
    _num("DistanceFromHome", "Distance from home", PERSONAL, "lives {} km from work", 7, 1, 29),
    _cat(
        "Department",
        "Department",
        JOB,
        "works in the {} department",
        ["Research & Development", "Sales", "Human Resources"],
        "Research & Development",
    ),
    _cat(
        "JobRole",
        "Job role",
        JOB,
        "works as a {}",
        [
            "Sales Executive",
            "Research Scientist",
            "Laboratory Technician",
            "Manufacturing Director",
            "Healthcare Representative",
            "Manager",
            "Sales Representative",
            "Research Director",
            "Human Resources",
        ],
        "Research Scientist",
    ),
    Feature(
        "BusinessTravel",
        "Business travel",
        JOB,
        _travel,
        "Travel_Rarely",
        choices={
            "Non-Travel": "Never",
            "Travel_Rarely": "Rarely",
            "Travel_Frequently": "Frequently",
        },
    ),
    Feature(
        "OverTime",
        "Works overtime",
        JOB,
        lambda v: "works overtime" if v == "Yes" else "does not work overtime",
        "No",
        choices={"No": "No", "Yes": "Yes"},
    ),
    Feature(
        "NumCompaniesWorked",
        "Previous companies",
        JOB,
        lambda v: f"has worked at {int(v)} {'company' if int(v) == 1 else 'companies'} before",
        2,
        minimum=0,
        maximum=9,
    ),
    _years("TotalWorkingYears", "Total working years", JOB, "has {} of experience", 10, 40),
    _years("YearsAtCompany", "Years at company", JOB, "has spent {} at the company", 5, 40),
    _years("YearsInCurrentRole", "Years in current role", JOB, "has {} in the current role", 3, 18),
    _years(
        "YearsSinceLastPromotion",
        "Years since last promotion",
        JOB,
        "was last promoted {} ago",
        1,
        15,
    ),
    _years(
        "YearsWithCurrManager", "Years with manager", JOB, "has {} with the current manager", 3, 17
    ),
    _num(
        "TrainingTimesLastYear",
        "Trainings last year",
        JOB,
        "attended {} trainings last year",
        3,
        0,
        6,
    ),
    _num("DailyRate", "Daily rate", PAY, "has a daily rate of {}", 800, 100, 1500),
    _num("HourlyRate", "Hourly rate", PAY, "has an hourly rate of {}", 65, 30, 100),
    _num("MonthlyRate", "Monthly rate", PAY, "has a monthly rate of {}", 14000, 2000, 27000),
    _num("PercentSalaryHike", "Last salary hike (%)", PAY, "received a {}% raise", 14, 11, 25),
    _num("StockOptionLevel", "Stock option level", PAY, "has stock option level {}", 1, 0, 3),
    _scale(
        "PerformanceRating",
        "Performance rating",
        SURVEY,
        "has a performance rating of {}",
        PERFORMANCE,
        3,
    ),
    _scale("JobInvolvement", "Job involvement", SURVEY, "shows {} job involvement", LEVEL4, 3),
    _scale(
        "EnvironmentSatisfaction",
        "Environment satisfaction",
        SURVEY,
        "reports {} environment satisfaction",
        LEVEL4,
        3,
    ),
    _scale(
        "RelationshipSatisfaction",
        "Relationship satisfaction",
        SURVEY,
        "reports {} relationship satisfaction",
        LEVEL4,
        3,
    ),
    _scale(
        "WorkLifeBalance", "Work-life balance", SURVEY, "has {} work-life balance", WORK_LIFE, 3
    ),
]

FEATURES_BY_NAME: dict[str, Feature] = {f.name: f for f in FEATURES}

# Columns the model predicts. They are never part of the input sentence.
TARGET_COLUMNS = ["Attrition", "JobLevel", "JobSatisfaction", "MonthlyIncome"]

# Constant or ID columns in the CSV; dropped during preparation.
DROPPED_COLUMNS = ["EmployeeCount", "EmployeeNumber", "Over18", "StandardHours"]

ALL_COLUMNS = [f.name for f in FEATURES] + TARGET_COLUMNS


def default_record() -> dict[str, Any]:
    """A plausible employee, used as the starting point in the web app."""
    return {f.name: f.default for f in FEATURES}


def validate_record(record: dict[str, Any]) -> dict[str, Any]:
    """Check a record and return it with numbers as int. Raises ValueError with a clear message."""
    clean: dict[str, Any] = {}
    for feature in FEATURES:
        if feature.name not in record or record[feature.name] is None:
            raise ValueError(f"missing value for '{feature.name}'")
        value = record[feature.name]
        if feature.is_categorical:
            if isinstance(value, float) and value.is_integer():
                value = int(value)
            if value not in feature.choices:
                allowed = ", ".join(map(str, feature.choices))
                raise ValueError(f"'{feature.name}' must be one of: {allowed} (got {value!r})")
        else:
            try:
                value = round(float(value))
            except (TypeError, ValueError):
                raise ValueError(f"'{feature.name}' must be a number (got {value!r})") from None
            if value < 0:
                raise ValueError(f"'{feature.name}' cannot be negative (got {value})")
        clean[feature.name] = value
    return clean


def record_to_text(record: dict[str, Any]) -> str:
    """Turn an employee record into the sentence the model reads.

    >>> record_to_text(default_record())[:60]
    'The employee is 35 years old, is male, is married, has bache'
    """
    record = validate_record(record)
    phrases = [f.phrase(record[f.name]) for f in FEATURES]
    return "The employee " + ", ".join(phrases) + "."
