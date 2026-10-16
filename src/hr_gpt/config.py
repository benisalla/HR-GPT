"""Every setting in one place: the tasks, the data, the model and the training run."""

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

CLASSIFICATION = "classification"
REGRESSION = "regression"

# Weights from the Hugging Face hub that can initialise the backbone
PRETRAINED = ("gpt2", "distilgpt2")


@dataclass(frozen=True)
class TaskSpec:
    """One prediction head."""

    column: str  # CSV column with the target
    kind: str  # CLASSIFICATION or REGRESSION
    classes: tuple[str, ...] = ()  # class names, in label order (classification)
    values: tuple = ()  # the CSV value of each class, same order (label = position)
    loss_weight: float = 1.0  # weight of this task in the total loss
    balanced: bool = False  # weight classes by 1 / frequency (for rare classes)
    log_target: bool = False  # regression: predict log(value) (for skewed values)

    @property
    def n_outputs(self) -> int:
        return len(self.classes) if self.kind == CLASSIFICATION else 1


TASKS: dict[str, TaskSpec] = {
    # 16% of employees leave: balanced weights stop the head from always saying "stays"
    "Attrition": TaskSpec(
        "Attrition",
        CLASSIFICATION,
        classes=("Stays", "Leaves"),
        values=("No", "Yes"),
        balanced=True,
    ),
    "JobLevel": TaskSpec(
        "JobLevel",
        CLASSIFICATION,
        classes=("Level 1", "Level 2", "Level 3", "Level 4", "Level 5"),
        values=(1, 2, 3, 4, 5),
    ),
    "JobSatisfaction": TaskSpec(
        "JobSatisfaction",
        CLASSIFICATION,
        classes=("Low", "Medium", "High", "Very High"),
        values=(1, 2, 3, 4),
    ),
    # Incomes range from 1k to 20k and are right-skewed: learn log(income), standardised
    "MonthlyIncome": TaskSpec("MonthlyIncome", REGRESSION, log_target=True),
}


@dataclass
class DataConfig:
    """Where the data lives and how it is split."""

    data_dir: Path = Path("data")
    csv_url: str = (
        "https://raw.githubusercontent.com/IBM/employee-attrition-aif360/"
        "master/data/emp_attrition.csv"
    )
    csv_md5: str = "ad8207459e5732574372cf8ff619883f"
    val_fraction: float = 0.1
    test_fraction: float = 0.1
    split_seed: int = 1337
    max_tokens: int = 512  # longest sentence is ~220 GPT-2 tokens

    @property
    def raw_csv(self) -> Path:
        return Path(self.data_dir) / "raw" / "hr_employee_attrition.csv"

    @property
    def processed_dir(self) -> Path:
        return Path(self.data_dir) / "processed"


@dataclass
class ModelConfig:
    """Transformer size. With `init` = gpt2/distilgpt2 the size comes from that checkpoint."""

    init: str = "gpt2"  # "gpt2" | "distilgpt2" | "scratch"
    vocab_size: int = 50258  # GPT-2's 50257 + <|pad|>
    block_size: int = 1024
    n_layer: int = 12
    n_head: int = 12
    n_embd: int = 768
    dropout: float = 0.1
    head_dropout: float = 0.1
    bias: bool = True

    def validate(self) -> "ModelConfig":
        if self.init not in (*PRETRAINED, "scratch"):
            raise ValueError(f"init must be one of {(*PRETRAINED, 'scratch')}, got {self.init!r}")
        if self.n_embd % self.n_head != 0:
            raise ValueError(f"n_embd ({self.n_embd}) must be divisible by n_head ({self.n_head})")
        if not 0 <= self.dropout < 1 or not 0 <= self.head_dropout < 1:
            raise ValueError("dropout values must be in [0, 1)")
        return self


@dataclass
class TrainConfig:
    """How long and how fast to train."""

    output_dir: Path = Path("checkpoints/run")
    epochs: int = 8
    batch_size: int = 16
    lr: float = 3e-5  # backbone (pretrained, so small)
    head_lr: float = 1e-3  # task heads (new, so larger)
    weight_decay: float = 0.01
    warmup_fraction: float = 0.1
    grad_clip: float = 1.0
    freeze_embeddings: bool = True  # keep GPT-2's 50k-token embedding table fixed (38.6M params)
    patience: int = 3  # stop after this many epochs without a better val loss (0 = never)
    seed: int = 1337
    device: str = "auto"  # "auto" | "cpu" | "cuda" | "mps"
    precision: str = "auto"  # "auto" (bf16 on supported GPUs, else fp32) | "fp32" | "bf16"
    log_every: int = 20  # steps
    num_workers: int = 0
    limit: int | None = None  # use only the first N training rows (quick checks)

    def validate(self) -> "TrainConfig":
        if self.epochs <= 0 or self.batch_size <= 0:
            raise ValueError("epochs and batch_size must be > 0")
        if self.lr <= 0 or self.head_lr <= 0:
            raise ValueError("learning rates must be > 0")
        if not 0 <= self.warmup_fraction < 1:
            raise ValueError("warmup_fraction must be in [0, 1)")
        if self.precision not in ("auto", "fp32", "bf16"):
            raise ValueError(f"precision must be auto, fp32 or bf16, got {self.precision!r}")
        return self


def to_jsonable(obj: Any) -> Any:
    """Dataclass -> dict with Paths as strings (for checkpoints and JSON files)."""
    data = asdict(obj) if hasattr(obj, "__dataclass_fields__") else obj
    if isinstance(data, dict):
        return {k: to_jsonable(v) for k, v in data.items()}
    if isinstance(data, (list, tuple)):
        return [to_jsonable(v) for v in data]
    if isinstance(data, Path):
        return str(data)
    return data


def tasks_from_dict(data: dict[str, dict[str, Any]]) -> dict[str, TaskSpec]:
    return {
        name: TaskSpec(
            **{**spec, "classes": tuple(spec["classes"]), "values": tuple(spec["values"])}
        )
        for name, spec in data.items()
    }
