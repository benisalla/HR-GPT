from dataclasses import dataclass, asdict, field
from typing import Dict, Optional, Any, Tuple

# spetial tokens 
SPECIAL_TOKENS = {
    "bos_token": "<sost>",
    'pad_token': "<pad>", 
}


# Task
@dataclass
class TaskSpec:
    target_col: str
    task_type: str                      # "binary", "multiclass", or "regression"
    n_classes: Optional[int] = None     # 1 implied for regression
    mult_fact: int = 4                  # width factor for the head MLP
    dropout: float = 0.2

@dataclass
class DatasetConfig:
    data_dir: str = "toy_data"
    toks_in_batch: int = 1000
    batch_size: int = 32
    data_stats_path: str = "data_stats.json"

# tasks
TASKS = {
    "Attrition": TaskSpec(
        target_col="Attrition",
        task_type="binary",
        n_classes=2,
        mult_fact=2,
        dropout=0.5,
    ),
    "JobLevel": TaskSpec(
        target_col="JobLevel",
        task_type="multiclass",
        n_classes=5,
        mult_fact=3,
        dropout=0.5,
    ),
    "JobSatisfaction": TaskSpec(
        target_col="JobSatisfaction",
        task_type="multiclass",
        n_classes=4,
        mult_fact=3,
        dropout=0.5,
    ),
    "MonthlyIncome": TaskSpec(
        target_col="MonthlyIncome",
        task_type="regression",
        mult_fact=4,
        dropout=0.5,
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
    tasks: Dict[str, TaskSpec] = field(default_factory=lambda: TASKS.copy())

    # dataset config 
    dataset: DatasetConfig = field(default_factory=DatasetConfig)

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
    

# Training configuration
@dataclass
class TrainConfig:
    # Dataloader / runtime
    num_workers: int = 4

    # Training length
    max_iters: Optional[int] = 10_000  

    # Optimizer
    learning_rate: float = 1e-4
    betas: Tuple[float, float] = (0.9, 0.95)
    weight_decay: float = 0.1
    grad_norm_clip: float = 1.0
    lbl_smoothing: float = 0.05

    # Logging & eval
    log_interval: int = 100        # steps
    eval_interval: int = 1000      # steps
    save_chpts_every : int = 1000  # steps
    log_dir: str = "logs"
    experiment_name: str = "hr_gpt_training"
    checkpoint_dir: str = "checkpoints"   

    # scaling 
    reg_loss_scale: float = 10.0
    cls_loss_scale: float = 0.1
    reg_unit_value: float = 1_000.0 # after training 1 => 1_000 ( unit value )

    # materials config
    device: str = "cpu"         
    amp: bool = True            

    # some utilities
    def validate(self) -> "TrainConfig":
        if self.batch_size <= 0:
            raise ValueError("batch_size must be > 0")
        if self.max_iters is not None and self.max_iters <= 0:
            raise ValueError("max_iters must be None or > 0")
        if self.learning_rate <= 0:
            raise ValueError("learning_rate must be > 0")
        if self.grad_norm_clip is not None and self.grad_norm_clip <= 0:
            raise ValueError("grad_norm_clip must be None or > 0")
        if self.log_interval <= 0:
            raise ValueError("log_interval must be > 0")
        if self.eval_interval <= 0:
            raise ValueError("eval_interval must be > 0")
        return self

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def update(self, **kwargs) -> "TrainConfig":
        for k, v in kwargs.items():
            if not hasattr(self, k):
                raise AttributeError(f"Unknown train config field: {k}")
            setattr(self, k, v)
        return self.validate()

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "TrainConfig":
        return cls(**d).validate()

    @property
    def log_every(self) -> int:
        return self.log_interval
    
    @property
    def val_every(self) -> int:
        return self.eval_interval