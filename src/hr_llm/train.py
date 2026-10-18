"""Training loop and the `hr-train` command.

Each epoch: train on every employee once, then measure the validation split.
`last.pt` is written every epoch, `best.pt` whenever the validation loss improves.
Training stops early after `patience` epochs without improvement.

    hr-train                                   # GPT-2 small, default settings
    hr-train --init distilgpt2 --epochs 6      # smaller and ~2x faster
    hr-train --limit 64 --epochs 2 --init scratch --layers 2 --embd 128 --heads 4   # smoke test
"""

import argparse
import json
import math
import random
import time
from collections.abc import Callable
from contextlib import nullcontext
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader

from hr_llm.config import TASKS, DataConfig, ModelConfig, TaskSpec, TrainConfig, to_jsonable
from hr_llm.data import (
    Collator,
    HRDataset,
    TargetEncoder,
    build_tokenizer,
    describe_columns,
    load_split,
)
from hr_llm.metrics import headline, task_metrics
from hr_llm.model import HRLLM, build_model
from hr_llm.predict import pick_device, run_model

BEST_CHECKPOINT = "best.pt"
LAST_CHECKPOINT = "last.pt"
TOKENIZER_DIR = "tokenizer"


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def lr_lambda(step: int, warmup_steps: int, total_steps: int) -> float:
    """LR multiplier: linear warmup to 1, then cosine decay to 0."""
    if step < warmup_steps:
        return (step + 1) / warmup_steps
    progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
    return 0.5 * (1.0 + math.cos(math.pi * min(1.0, progress)))


def autocast_for(device: torch.device, precision: str):
    """bf16 autocast on GPUs that support it (no loss scaling needed), else plain fp32."""
    use_bf16 = precision == "bf16" or (
        precision == "auto" and device.type == "cuda" and torch.cuda.is_bf16_supported()
    )
    if use_bf16 and device.type in ("cuda", "cpu"):
        return torch.autocast(device.type, dtype=torch.bfloat16)
    return nullcontext()


@torch.no_grad()
def evaluate(
    model: HRLLM,
    dataset: HRDataset,
    targets: TargetEncoder,
    pad_token_id: int,
    class_weights: dict[str, torch.Tensor | None],
    batch_size: int = 32,
) -> tuple[float, dict[str, dict[str, float]]]:
    """Validation loss (same weighting as training) and metrics per task."""
    model.eval()
    device = next(model.parameters()).device
    collate = Collator(pad_token_id)
    total, n = 0.0, 0
    for start in range(0, len(dataset), batch_size):
        batch = collate([dataset[i] for i in range(start, min(start + batch_size, len(dataset)))])
        outputs = model(batch["input_ids"].to(device), batch["lengths"].to(device))
        loss, _ = model.loss(outputs, batch["targets"], class_weights)
        total += loss.item() * len(batch["lengths"])
        n += len(batch["lengths"])
    preds = run_model(model, dataset.ids, pad_token_id, targets, batch_size)
    metrics = {}
    for name, spec in model.tasks.items():
        if spec.kind == "classification":
            y_true = np.array([t[name] for t in dataset.targets])
        else:
            y_true = np.array([r[spec.column] for r in dataset.records], dtype=float)
        metrics[name] = task_metrics(spec, y_true, preds[name])
    return total / max(1, n), metrics


def save_checkpoint(path: Path, model: HRLLM, extra: dict[str, Any]) -> None:
    tmp = path.with_suffix(".tmp")
    torch.save({"model": model.state_dict(), **extra}, tmp)
    tmp.replace(path)  # never leave a half-written checkpoint


def format_metrics(tasks: dict[str, TaskSpec], metrics: dict[str, dict[str, float]]) -> str:
    parts = []
    for name, spec in tasks.items():
        m = metrics[name]
        if spec.kind == "classification":
            extra = f" auc={m['roc_auc']:.3f}" if "roc_auc" in m else ""
            parts.append(f"{name}: bal_acc={m['balanced_accuracy']:.3f}{extra}")
        else:
            parts.append(f"{name}: mae=${m['mae']:,.0f} r2={m['r2']:.3f}")
    return " | ".join(parts)


def train(
    data_config: DataConfig,
    model_config: ModelConfig,
    train_config: TrainConfig,
    tasks: dict[str, TaskSpec] | None = None,
    tokenizer=None,
    model: HRLLM | None = None,
    on_epoch_end: Callable[[list[dict[str, Any]]], None] | None = None,
) -> dict[str, Any]:
    """Train and return a summary (best epoch, best val loss, history). `tokenizer`/`model`
    can be passed in (tests use tiny ones); otherwise GPT-2's tokenizer and `build_model` are used.
    `on_epoch_end(history)` is called after each epoch (the notebook draws its dashboard there).
    """
    train_config.validate()
    tasks = dict(tasks or TASKS)
    seed_everything(train_config.seed)
    device = pick_device(train_config.device)
    out = Path(train_config.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    # data
    train_frame = load_split(data_config, "train")
    if train_config.limit:
        train_frame = train_frame.iloc[: train_config.limit]
    val_frame = load_split(data_config, "val")
    tokenizer = tokenizer or build_tokenizer("gpt2")
    targets = TargetEncoder.fit(tasks, train_frame)
    class_weights = {name: targets.class_weights(name, train_frame) for name in tasks}
    train_set = HRDataset(train_frame, tokenizer, targets, data_config.max_tokens)
    val_set = HRDataset(val_frame, tokenizer, targets, data_config.max_tokens)
    pad_id = tokenizer.pad_token_id
    loader = DataLoader(
        train_set,
        batch_size=train_config.batch_size,
        shuffle=True,
        collate_fn=Collator(pad_id),
        num_workers=train_config.num_workers,
        generator=torch.Generator().manual_seed(train_config.seed),
    )
    print(describe_columns())
    print(f"train={len(train_set)} val={len(val_set)} employees | device={device}")

    # model + optimisation
    model_config.vocab_size = len(tokenizer)
    model = (model or build_model(model_config, tasks)).to(device)
    if train_config.freeze_embeddings:
        model.transformer.wte.weight.requires_grad_(False)
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(
        f"model: init={model.config.init} layers={model.config.n_layer} "
        f"d={model.config.n_embd} params={model.num_parameters() / 1e6:.1f}M "
        f"(trainable {trainable / 1e6:.1f}M)"
    )
    optimizer = AdamW(
        model.param_groups(train_config.lr, train_config.head_lr, train_config.weight_decay)
    )
    total_steps = train_config.epochs * len(loader)
    warmup = max(1, int(train_config.warmup_fraction * total_steps))
    scheduler = LambdaLR(optimizer, lambda s: lr_lambda(s, warmup, total_steps))
    tokenizer.save_pretrained(out / TOKENIZER_DIR)
    autocast = autocast_for(device, train_config.precision)

    meta = {
        "model_config": to_jsonable(model.config),
        "tasks": to_jsonable(tasks),
        "targets": targets.state_dict(),
        "max_tokens": data_config.max_tokens,
        "tokenizer_dir": TOKENIZER_DIR,
        "data_config": to_jsonable(data_config),
        "train_config": to_jsonable(train_config),
    }
    history: list[dict[str, Any]] = []
    best_loss, best_epoch, bad_epochs, step = float("inf"), -1, 0, 0
    start = time.time()

    for epoch in range(1, train_config.epochs + 1):
        model.train()
        running, seen = 0.0, 0
        for batch in loader:
            with autocast:
                outputs = model(batch["input_ids"].to(device), batch["lengths"].to(device))
            loss, parts = model.loss(outputs, batch["targets"], class_weights)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), train_config.grad_clip)
            optimizer.step()
            scheduler.step()
            step += 1
            running += loss.item() * len(batch["lengths"])
            seen += len(batch["lengths"])
            if step % train_config.log_every == 0:
                detail = " ".join(f"{k}={v:.3f}" for k, v in parts.items())
                print(
                    f"epoch {epoch} step {step}/{total_steps} loss={loss.item():.4f} "
                    f"({detail}) lr={scheduler.get_last_lr()[0]:.2e} {time.time() - start:.0f}s"
                )

        val_loss, val_metrics = evaluate(model, val_set, targets, pad_id, class_weights)
        row = {
            "epoch": epoch,
            "step": step,
            "train_loss": running / max(1, seen),
            "val_loss": val_loss,
            "val": val_metrics,
            "val_headline": {n: headline(s, val_metrics[n]) for n, s in tasks.items()},
            "seconds": round(time.time() - start, 1),
        }
        history.append(row)
        print(f"== epoch {epoch}: train_loss={row['train_loss']:.4f} val_loss={val_loss:.4f}")
        print(f"   {format_metrics(tasks, val_metrics)}")

        info = {"epoch": epoch, "step": step, "val_loss": val_loss, "val_metrics": val_metrics}
        save_checkpoint(out / LAST_CHECKPOINT, model, {**meta, **info})
        if val_loss < best_loss:
            best_loss, best_epoch, bad_epochs = val_loss, epoch, 0
            save_checkpoint(out / BEST_CHECKPOINT, model, {**meta, **info})
            print(f"   new best -> {out / BEST_CHECKPOINT}")
        else:
            bad_epochs += 1
        (out / "history.json").write_text(json.dumps(history, indent=2))
        if on_epoch_end:
            on_epoch_end(history)
        if train_config.patience and bad_epochs >= train_config.patience:
            print(f"stopping early: no improvement for {bad_epochs} epochs")
            break

    summary = {"best_epoch": best_epoch, "best_val_loss": best_loss, "history": history}
    print(f"done in {time.time() - start:.0f}s; best epoch {best_epoch} (val_loss={best_loss:.4f})")
    return summary


def build_parser() -> argparse.ArgumentParser:
    d, m, t = DataConfig(), ModelConfig(), TrainConfig()
    p = argparse.ArgumentParser(description="Train HR-LLM on the IBM HR attrition data.")
    p.add_argument("--data-dir", type=Path, default=d.data_dir)
    p.add_argument("--output-dir", type=Path, default=t.output_dir)
    p.add_argument(
        "--init",
        default=m.init,
        choices=["gpt2", "distilgpt2", "scratch"],
        help="start from GPT-2, DistilGPT-2 or random weights",
    )
    p.add_argument("--layers", type=int, default=m.n_layer, help="only with --init scratch")
    p.add_argument("--heads", type=int, default=m.n_head, help="only with --init scratch")
    p.add_argument("--embd", type=int, default=m.n_embd, help="only with --init scratch")
    p.add_argument("--dropout", type=float, default=m.dropout)
    p.add_argument("--epochs", type=int, default=t.epochs)
    p.add_argument("--batch-size", type=int, default=t.batch_size)
    p.add_argument("--lr", type=float, default=t.lr, help="backbone learning rate")
    p.add_argument("--head-lr", type=float, default=t.head_lr, help="task heads learning rate")
    p.add_argument("--weight-decay", type=float, default=t.weight_decay)
    p.add_argument("--patience", type=int, default=t.patience)
    p.add_argument("--seed", type=int, default=t.seed)
    p.add_argument("--device", default=t.device)
    p.add_argument("--precision", default=t.precision, choices=["auto", "fp32", "bf16"])
    p.add_argument("--log-every", type=int, default=t.log_every)
    p.add_argument(
        "--train-embeddings",
        action="store_true",
        help="also fine-tune the token embedding table (frozen by default)",
    )
    p.add_argument("--limit", type=int, default=None, help="train on the first N rows only")
    return p


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    data_config = DataConfig(data_dir=args.data_dir)
    model_config = ModelConfig(
        init=args.init,
        n_layer=args.layers,
        n_head=args.heads,
        n_embd=args.embd,
        dropout=args.dropout,
    )
    train_config = TrainConfig(
        output_dir=args.output_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        head_lr=args.head_lr,
        weight_decay=args.weight_decay,
        patience=args.patience,
        seed=args.seed,
        device=args.device,
        precision=args.precision,
        log_every=args.log_every,
        limit=args.limit,
        freeze_embeddings=not args.train_embeddings,
    )
    train(data_config, model_config, train_config)


if __name__ == "__main__":
    main()
