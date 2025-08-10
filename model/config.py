from dataclasses import dataclass, asdict
from typing import Dict, Optional, Any

# Task
@dataclass
class TaskSpec:
    target_col: str
    task_type: str                      # "binary", "multiclass", or "regression"
    n_classes: Optional[int] = None     # required for classification; 1 implied for regression
    mult_fact: int = 4                  # width factor for the head MLP
    dropout: float = 0.2


# drop constants/IDs
COMMON_DROP = ["EmployeeNumber", "EmployeeCount", "Over18", "StandardHours"]

# tasks
TASKS = {
    "Attrition": TaskSpec(
        target_col="Attrition",
        task_type="binary",
        n_classes=2,
        mult_fact=4,
        dropout=0.2,
    ),

    "JobLevel": TaskSpec(
        target_col="JobLevel",
        task_type="multiclass",
        n_classes=5,
        mult_fact=4,
        dropout=0.2,
    ),

    "MonthlyIncome": TaskSpec(
        target_col="MonthlyIncome",
        task_type="regression",
        mult_fact=4,
        dropout=0.2,
    ),

    "JobSatisfaction": TaskSpec(
        target_col="JobSatisfaction",
        task_type="multiclass",
        n_classes=4,
        mult_fact=4,
        dropout=0.2,
    ),
}

# Main model config
@dataclass
class GPTConfig:
    # required
    vocab_size: int = 50257
    block_size: int = 1024

    # model dims
    n_layer: int = 12
    n_head: int  = 12
    n_embd: int  = 768

    # misc
    bias: bool = True
    embd_pdrop: float = 0.1
    resid_pdrop: float = 0.1
    attn_pdrop: float  = 0.1

    # tasks
    tasks: Dict[str, TaskSpec] = TASKS

    def validate(self) -> "GPTConfig":
        if self.n_embd % self.n_head != 0:
            raise ValueError("n_embd must be divisible by n_head.")
        
        # sanity checks on tasks
        for name, t in self.tasks.items():
            if t.task_type in ("binary", "multiclass"):
                if t.n_classes <= 0:
                    raise ValueError(f"{name}: n_classes must be greater than 0 for classification tasks.")
                if t.task_type == "binary" and t.n_classes != 2:
                    raise ValueError(f"{name}: binary tasks must have n_classes=2.")
            elif t.task_type == "regression":
                if t.n_classes not in (None, 1):
                    raise ValueError(f"{name}: regression tasks must have n_classes=None or 1.")
            else:
                raise ValueError(f"{name}: unknown task_type={t.task_type}")
        return self

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def update(self, **kwargs) -> "GPTConfig":
        for k, v in kwargs.items():
            if not hasattr(self, k):
                raise AttributeError(f"Unknown config field: {k}")
            setattr(self, k, v)
        return self.validate()

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "GPTConfig":
        return cls(**d).validate()