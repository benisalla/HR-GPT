from typing import Dict, Any, List
from copy import deepcopy

# drop constants/IDs
COMMON_DROP = ["EmployeeNumber", "EmployeeCount", "Over18", "StandardHours"]

# tasks attributes
TASKS_ATTRS = [
    "Attrition", "JobLevel", "MonthlyIncome", "JobSatisfaction"
]

# ordered attributes
ORDERED_ATTRS = [
 'Age','Attrition','BusinessTravel','DailyRate','Department','DistanceFromHome',
 'Education','EducationField','EnvironmentSatisfaction','Gender','HourlyRate',
 'JobInvolvement','JobLevel','JobRole','JobSatisfaction','MaritalStatus',
 'MonthlyIncome','MonthlyRate','NumCompaniesWorked','OverTime','PercentSalaryHike',
 'PerformanceRating','RelationshipSatisfaction','StockOptionLevel','TotalWorkingYears',
 'TrainingTimesLastYear','WorkLifeBalance','YearsAtCompany','YearsInCurrentRole',
 'YearsSinceLastPromotion','YearsWithCurrManager'
]

# String categoricals
CATEGORICAL_STR_COLS = [
    "BusinessTravel","Department","EducationField","Gender","JobRole","MaritalStatus","OverTime",
]

# Pure numeric columns
NUMERIC_COLS = [
    "Age","DailyRate","DistanceFromHome","Education","EnvironmentSatisfaction","HourlyRate",
    "JobInvolvement","JobLevel","JobSatisfaction","MonthlyIncome","MonthlyRate",
    "NumCompaniesWorked","PercentSalaryHike","PerformanceRating","RelationshipSatisfaction",
    "StockOptionLevel","TotalWorkingYears","TrainingTimesLastYear","WorkLifeBalance",
    "YearsAtCompany","YearsInCurrentRole","YearsSinceLastPromotion","YearsWithCurrManager",
]

# Mappings for pretty printing
EDUCATION_MAP = {0: "below college", 1: "college", 2: "bachelor's", 3: "master's", 4: "doctorate"}
LEVEL4_MAP = {0: "low", 1: "medium", 2: "high", 3: "very high"}  
PERF_MAP = {0: "low", 1: "good", 2: "excellent", 3: "outstanding"}  
GENDER_VALUES = {"male", "female"}
YESNO_VALUES = {"yes", "no"}
WORKLIFEBALANCE_MAP = {0: "bad", 1: "good", 2: "better", 3: "best"}

def _plural(n: int, one: str, many: str = None) -> str:
    many = many or (one + "s")
    return one if int(n) == 1 else many

def _to_int(x):
    try:
        return int(x)
    except Exception:
        return x

def _to_float(x):
    try:
        return float(x)
    except Exception:
        return x

def _norm_str(x):
    return str(x).strip()

def _map01(m, val):
    n = _to_int(val)
    if isinstance(n, int):
        return m.get(n, m.get(n-1, str(val)))  # try 0-based, else 1-based
    return str(val)

def _pretty_attr(attr: str, val) -> str:
    val_s = _norm_str(val)
    val_l = val_s.lower()

    if attr == "Age":
        n = _to_int(val)
        return f"is {n} years old"

    if attr == "Gender":
        if val_l in GENDER_VALUES:
            return f"is {val_l}"
        return f"gender is {val_s}"

    if attr == "MaritalStatus":
        return f"is {val_l}"
    if attr == "Department":
        return f"works in the {val_s} department"
    if attr == "JobRole":
        return f"works as a {val_s}"
    if attr == "Education":
        return f"has {_map01(EDUCATION_MAP, val)} education"
    if attr == "EducationField":
        return f"studied {val_s}"
    if attr == "JobLevel":
        return f"is at job level {_to_int(val)}"
    if attr == "BusinessTravel":
        if "non" in val_l:
            return "does not travel for work"
        if "frequent" in val_l:
            return "travels frequently for work"
        if "rare" in val_l:
            return "travels rarely for work"
        return f"business travel: {val_s}"

    if attr == "OverTime":
        if val_l in YESNO_VALUES:
            return "works overtime" if val_l == "yes" else "does not work overtime"
        try:
            return "works overtime" if int(val) == 1 else "does not work overtime"
        except Exception:
            return f"overtime: {val_s}"

    if attr == "Attrition":
        if val_l in YESNO_VALUES:
            return "is leaving the company" if val_l == "yes" else "is staying with the company"
        try:
            return "is leaving the company" if int(val) == 1 else "is staying with the company"
        except Exception:
            return f"attrition: {val_s}"

    if attr == "DistanceFromHome":
        return f"has a commute distance of {_to_int(val)}"
    if attr == "MonthlyIncome":
        return f"earns {_to_int(val)} per month"
    if attr == "HourlyRate":
        return f"has an hourly rate of {_to_int(val)}"
    if attr == "DailyRate":
        return f"has a daily rate of {_to_int(val)}"
    if attr == "MonthlyRate":
        return f"has a monthly rate of {_to_int(val)}"
    if attr == "PercentSalaryHike":
        return f"received a {_to_int(val)}% raise"
    if attr == "PerformanceRating":
        return f"has a {_map01(PERF_MAP, val)} performance rating"
    if attr == "EnvironmentSatisfaction":
        return f"reports {_map01(LEVEL4_MAP, val)} environment satisfaction"
    if attr == "JobInvolvement":
        return f"shows {_map01(LEVEL4_MAP, val)} job involvement"
    if attr == "JobSatisfaction":
        return f"has {_map01(LEVEL4_MAP, val)} job satisfaction"
    if attr == "RelationshipSatisfaction":
        return f"has {_map01(LEVEL4_MAP, val)} relationship satisfaction"
    if attr == "WorkLifeBalance":
        return f"has {_map01(WORKLIFEBALANCE_MAP, val)} work–life balance"

    if attr == "NumCompaniesWorked":
        n = _to_int(val)
        return f"has worked at {n} {_plural(n, 'company')}"

    if attr == "TotalWorkingYears":
        n = _to_int(val)
        return f"has {n} {_plural(n, 'year')} of experience"

    if attr == "TrainingTimesLastYear":
        n = _to_int(val)
        return f"attended {n} {_plural(n, 'training')} last year"

    if attr == "YearsAtCompany":
        n = _to_int(val)
        return f"has spent {n} {_plural(n, 'year')} at the company"

    if attr == "YearsSinceLastPromotion":
        n = _to_int(val)
        return f"was last promoted {n} {_plural(n, 'year')} ago"
    
    if attr == "YearsInCurrentRole":
        n = _to_int(val)
        return f"has {n} {_plural(n, 'year')} in the current role"

    if attr == "YearsWithCurrManager":
        n = _to_int(val)
        return f"has {n} {_plural(n, 'year')} with the current manager"

    if attr == "StockOptionLevel":
        n = _to_int(val)
        return f"has stock option level {n}"

    return f"{attr} is {val_s}"

ATTRIBUTE_INPUT_FIELDS: Dict[str, Dict[str, Any]] = {
    "Age": {"type": "number", "label": "Age", "value": 30, "minimum": 18, "maximum": 70},
    "BusinessTravel": {"type": "dropdown", "label": "Business Travel", 
                        "choices": ["non-travel", "travel_rarely", "travel_frequently"], "value": "travel_rarely"},
    "DailyRate": {"type": "number", "label": "Daily Rate ($)", "value": 800, "minimum": 100, "maximum": 2000},
    "Department": {"type": "dropdown", "label": "Department",
                    "choices": ["sales", "research & development", "human resources"], "value": "research & development"},
    "DistanceFromHome": {"type": "number", "label": "Distance from Home (km)", "value": 10, "minimum": 1, "maximum": 50},
    "Education": {"type": "dropdown", "label": "Education Level",
                "choices": ["Below College", "College", "Bachelor", "Master", "Doctor"], "value": "Bachelor"},
    "EducationField": {"type": "dropdown", "label": "Education Field",
                        "choices": ["life sciences", "medical", "marketing", "technical degree", "other", "human resources"], 
                        "value": "life sciences"},
    "EnvironmentSatisfaction": {"type": "dropdown", "label": "Environment Satisfaction",
                                "choices": ["Low", "Medium", "High", "Very High"], "value": "High"},
    "Gender": {"type": "dropdown", "label": "Gender", "choices": ["male", "female"], "value": "male"},
    "HourlyRate": {"type": "number", "label": "Hourly Rate ($)", "value": 50, "minimum": 20, "maximum": 100},
    "JobInvolvement": {"type": "dropdown", "label": "Job Involvement",
                        "choices": ["Low", "Medium", "High", "Very High"], "value": "High"},
    "JobRole": {"type": "dropdown", "label": "Job Role",
                "choices": ["sales executive", "research scientist", "laboratory technician", 
                        "manufacturing director", "healthcare representative", "manager", 
                        "sales representative", "research director", "human resources"], 
                "value": "research scientist"},
    "MaritalStatus": {"type": "dropdown", "label": "Marital Status",
                    "choices": ["single", "married", "divorced"], "value": "married"},
    "MonthlyRate": {"type": "number", "label": "Monthly Rate ($)", "value": 15000, "minimum": 2000, "maximum": 30000},
    "NumCompaniesWorked": {"type": "number", "label": "Number of Companies Worked", "value": 2, "minimum": 0, "maximum": 10},
    "OverTime": {"type": "dropdown", "label": "Works Overtime", "choices": ["no", "yes"], "value": "no"},
    "PercentSalaryHike": {"type": "number", "label": "Percent Salary Hike (%)", "value": 15, "minimum": 10, "maximum": 25},
    "PerformanceRating": {"type": "dropdown", "label": "Performance Rating",
                        "choices": ["Low", "Good", "Excellent", "Outstanding"], "value": "Excellent"},
    "RelationshipSatisfaction": {"type": "dropdown", "label": "Relationship Satisfaction",
                                "choices": ["Low", "Medium", "High", "Very High"], "value": "High"},
    "StockOptionLevel": {"type": "number", "label": "Stock Option Level", "value": 1, "minimum": 0, "maximum": 3},
    "TotalWorkingYears": {"type": "number", "label": "Total Working Years", "value": 8, "minimum": 0, "maximum": 40},
    "TrainingTimesLastYear": {"type": "number", "label": "Training Times Last Year", "value": 3, "minimum": 0, "maximum": 10},
    "WorkLifeBalance": {"type": "dropdown", "label": "Work-Life Balance",
                        "choices": ["Bad", "Good", "Better", "Best"], "value": "Better"},
    "YearsAtCompany": {"type": "number", "label": "Years at Company", "value": 5, "minimum": 0, "maximum": 40},
    "YearsInCurrentRole": {"type": "number", "label": "Years in Current Role", "value": 3, "minimum": 0, "maximum": 20},
    "YearsSinceLastPromotion": {"type": "number", "label": "Years Since Last Promotion", "value": 1, "minimum": 0, "maximum": 15},
    "YearsWithCurrManager": {"type": "number", "label": "Years with Current Manager", "value": 2, "minimum": 0, "maximum": 15}
}

ENCODE_AS_INT: List[str] = [
    "Education",
    "EnvironmentSatisfaction",
    "JobInvolvement",
    "PerformanceRating",
    "RelationshipSatisfaction",
    "WorkLifeBalance",
]

ATTRIBUTE_CATEGORICAL_ENCODINGS: Dict[str, Dict[str, int]] = {
    field: {label: idx for idx, label in enumerate(ATTRIBUTE_INPUT_FIELDS[field]["choices"])}
    for field in ENCODE_AS_INT
}

CLASS_MAPPINGS: Dict[str, Dict[int, str]] = {
    "Attrition": {0: "No (Staying)", 1: "Yes (Leaving)"},
    "JobLevel": {0: "Level 1", 1: "Level 2", 2: "Level 3", 3: "Level 4", 4: "Level 5"},
    "JobSatisfaction": {0: "Low", 1: "Medium", 2: "High", 3: "Very High"},
}

def get_attribute_input_fields() -> Dict[str, Dict[str, Any]]:
    return deepcopy(ATTRIBUTE_INPUT_FIELDS)
