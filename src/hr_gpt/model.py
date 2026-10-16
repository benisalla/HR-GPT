"""HR-GPT: a GPT-2 style Transformer that reads an employee sentence and feeds 4 task heads.

    "The employee is 41 years old, ..." <eos>
        -> token + position embeddings
        -> N causal Transformer blocks
        -> hidden state at <eos> (it has attended to the whole sentence)
        -> one small MLP head per task: Attrition, JobLevel, JobSatisfaction, MonthlyIncome

The backbone can start from the real GPT-2 (or DistilGPT-2) weights, which is what
makes this work on only ~1.2k training rows.
"""

import math

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from hr_gpt.config import CLASSIFICATION, PRETRAINED, ModelConfig, TaskSpec

INIT_STD = 0.02  # GPT-2 initialisation


class CausalSelfAttention(nn.Module):
    """Multi-head self-attention where each token sees only itself and earlier tokens."""

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.n_head = config.n_head
        self.dropout = config.dropout
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd, bias=config.bias)  # q, k, v
        self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        self.resid_dropout = nn.Dropout(config.dropout)

    def forward(self, x: Tensor) -> Tensor:
        B, T, C = x.shape
        q, k, v = self.c_attn(x).split(C, dim=2)
        # [B, T, C] -> [B, heads, T, C / heads]
        q, k, v = (t.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) for t in (q, k, v))
        y = F.scaled_dot_product_attention(
            q, k, v, is_causal=True, dropout_p=self.dropout if self.training else 0.0
        )
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        return self.resid_dropout(self.c_proj(y))


class MLP(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd, bias=config.bias)
        self.gelu = nn.GELU(approximate="tanh")  # the GELU variant GPT-2 was trained with
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd, bias=config.bias)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x: Tensor) -> Tensor:
        return self.dropout(self.c_proj(self.gelu(self.c_fc(x))))


class Block(nn.Module):
    """Pre-norm Transformer block: x + attn(ln(x)), then x + mlp(ln(x))."""

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.n_embd, bias=config.bias)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = nn.LayerNorm(config.n_embd, bias=config.bias)
        self.mlp = MLP(config)

    def forward(self, x: Tensor) -> Tensor:
        x = x + self.attn(self.ln_1(x))
        return x + self.mlp(self.ln_2(x))


class TaskHead(nn.Module):
    """[B, n_embd] -> [B, n_outputs]: Linear -> GELU -> Dropout -> Linear."""

    def __init__(self, n_embd: int, n_outputs: int, dropout: float):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, n_embd),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(n_embd, n_outputs),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.net(x)


class HRGPT(nn.Module):
    def __init__(self, config: ModelConfig, tasks: dict[str, TaskSpec]):
        super().__init__()
        self.config = config.validate()
        self.tasks = dict(tasks)
        self.transformer = nn.ModuleDict(
            {
                "wte": nn.Embedding(config.vocab_size, config.n_embd),
                "wpe": nn.Embedding(config.block_size, config.n_embd),
                "drop": nn.Dropout(config.dropout),
                "h": nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
                "ln_f": nn.LayerNorm(config.n_embd, bias=config.bias),
            }
        )
        self.heads = nn.ModuleDict(
            {
                name: TaskHead(config.n_embd, spec.n_outputs, config.head_dropout)
                for name, spec in tasks.items()
            }
        )
        self.apply(self._init_weights)
        # GPT-2: scale the residual projections by 1/sqrt(2 * n_layer)
        for name, p in self.named_parameters():
            if name.endswith("c_proj.weight"):
                nn.init.normal_(p, mean=0.0, std=INIT_STD / math.sqrt(2 * config.n_layer))

    @staticmethod
    def _init_weights(module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=INIT_STD)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=INIT_STD)

    # -- forward ------------------------------------------------------------

    def encode(self, input_ids: Tensor, lengths: Tensor) -> Tensor:
        """Token ids [B, T] -> one vector per sentence [B, n_embd], read at the last real token."""
        B, T = input_ids.shape
        if T > self.config.block_size:
            raise ValueError(f"sequence length {T} > block_size {self.config.block_size}")
        pos = torch.arange(T, device=input_ids.device)
        x = self.transformer.drop(self.transformer.wte(input_ids) + self.transformer.wpe(pos))
        for block in self.transformer.h:
            x = block(x)
        x = self.transformer.ln_f(x)
        last = (lengths - 1).to(input_ids.device)  # padding is at the end, after this token
        return x[torch.arange(B, device=input_ids.device), last]

    def forward(self, input_ids: Tensor, lengths: Tensor) -> dict[str, Tensor]:
        """Raw outputs per task: logits [B, n_classes] or a standardised value [B]."""
        h = self.encode(input_ids, lengths)
        out = {}
        for name, head in self.heads.items():
            y = head(h)
            out[name] = y if self.tasks[name].kind == CLASSIFICATION else y.squeeze(-1)
        return out

    def loss(
        self,
        outputs: dict[str, Tensor],
        targets: dict[str, Tensor],
        class_weights: dict[str, Tensor | None] | None = None,
    ) -> tuple[Tensor, dict[str, float]]:
        """Weighted sum of per-task losses: cross-entropy (classification), MSE (regression)."""
        class_weights = class_weights or {}
        total = outputs[next(iter(outputs))].new_zeros(())
        parts: dict[str, float] = {}
        for name, spec in self.tasks.items():
            y = targets[name].to(outputs[name].device)
            if spec.kind == CLASSIFICATION:
                w = class_weights.get(name)
                loss = F.cross_entropy(
                    outputs[name].float(), y, weight=None if w is None else w.to(y.device)
                )
            else:
                loss = F.mse_loss(outputs[name].float(), y)
            total = total + spec.loss_weight * loss
            parts[name] = loss.item()
        return total, parts

    # -- helpers ------------------------------------------------------------

    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def param_groups(self, lr: float, head_lr: float, weight_decay: float) -> list[dict]:
        """AdamW groups: backbone vs heads (different LR), decay only on matrices."""
        groups: dict[tuple[bool, bool], list[nn.Parameter]] = {}
        for name, p in self.named_parameters():
            if not p.requires_grad:
                continue
            is_head = name.startswith("heads.")
            decay = p.dim() >= 2 and "wte" not in name and "wpe" not in name
            groups.setdefault((is_head, decay), []).append(p)
        return [
            {
                "params": params,
                "lr": head_lr if is_head else lr,
                "weight_decay": weight_decay if decay else 0.0,
            }
            for (is_head, decay), params in groups.items()
        ]


def config_for_pretrained(name: str, base: ModelConfig) -> ModelConfig:
    """Copy the architecture of a Hugging Face GPT-2 checkpoint into our config."""
    from transformers import AutoConfig

    hf = AutoConfig.from_pretrained(name)
    return ModelConfig(
        init=name,
        vocab_size=base.vocab_size,
        block_size=hf.n_positions,
        n_layer=hf.n_layer,
        n_head=hf.n_head,
        n_embd=hf.n_embd,
        dropout=base.dropout,
        head_dropout=base.head_dropout,
        bias=True,
    )


@torch.no_grad()
def load_pretrained_backbone(model: HRGPT, name: str) -> int:
    """Copy GPT-2 / DistilGPT-2 weights into the backbone. Returns the number of tensors copied.

    Hugging Face stores the attention/MLP matrices as Conv1D ([in, out]); ours are
    nn.Linear ([out, in]), so those are transposed. The extra <|pad|> embedding row
    gets the mean of GPT-2's embeddings. The task heads stay random.
    """
    from transformers import GPT2Model

    if name not in PRETRAINED:
        raise ValueError(f"unknown pretrained model {name!r}, expected one of {PRETRAINED}")
    hf = GPT2Model.from_pretrained(name).state_dict()
    ours = model.state_dict()
    transposed = (
        "attn.c_attn.weight",
        "attn.c_proj.weight",
        "mlp.c_fc.weight",
        "mlp.c_proj.weight",
    )
    copied = 0
    for key, value in hf.items():
        target = f"transformer.{key}"
        if target not in ours:
            continue  # e.g. the attention mask buffers of old transformers versions
        if key == "wte.weight":
            n = value.shape[0]
            ours[target][:n].copy_(value)
            ours[target][n:].copy_(value.mean(dim=0, keepdim=True))
        elif key.endswith(transposed):
            ours[target].copy_(value.t())
        else:
            ours[target].copy_(value)
        copied += 1
    expected = sum(1 for k in ours if k.startswith("transformer."))
    if copied != expected:
        raise RuntimeError(f"copied {copied} of {expected} backbone tensors from {name}")
    return copied


def build_model(
    config: ModelConfig, tasks: dict[str, TaskSpec], load_weights: bool = True
) -> HRGPT:
    """Create the model; with config.init = gpt2/distilgpt2, size and (optionally) weights come from it."""
    config = config.validate()
    if config.init in PRETRAINED:
        config = config_for_pretrained(config.init, config)
    model = HRGPT(config, tasks)
    if load_weights and config.init in PRETRAINED:
        load_pretrained_backbone(model, config.init)
    return model
