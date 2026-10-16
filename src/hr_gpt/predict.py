"""Load a trained checkpoint and predict all four tasks for one or many employees.

Python:
    from hr_gpt.predict import Predictor
    predictor = Predictor.from_checkpoint("checkpoints/run/best.pt")
    predictor.predict(record)              # one employee (dict of the 27 input columns)

Command line:
    hr-predict --checkpoint checkpoints/run/best.pt --json employee.json
    hr-predict --checkpoint checkpoints/run/best.pt --test-row 0
"""

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch

from hr_gpt.config import CLASSIFICATION, DataConfig, ModelConfig, TaskSpec, tasks_from_dict
from hr_gpt.data import (
    Collator,
    TargetEncoder,
    build_tokenizer,
    encode_text,
    frame_to_records,
    load_split,
)
from hr_gpt.features import TARGET_COLUMNS, record_to_text, validate_record
from hr_gpt.model import HRGPT


def pick_device(name: str = "auto") -> torch.device:
    if name != "auto":
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


@torch.no_grad()
def run_model(
    model: HRGPT,
    ids: list[list[int]],
    pad_token_id: int,
    targets: TargetEncoder,
    batch_size: int = 32,
) -> dict[str, np.ndarray]:
    """Token ids of N sentences -> {task: probs [N, C] or values in real units [N]}."""
    model.eval()
    device = next(model.parameters()).device
    collate = Collator(pad_token_id)
    chunks: dict[str, list[np.ndarray]] = {name: [] for name in model.tasks}
    for start in range(0, len(ids), batch_size):
        batch = collate([{"input_ids": x, "targets": {}} for x in ids[start : start + batch_size]])
        outputs = model(batch["input_ids"].to(device), batch["lengths"].to(device))
        for name, out in outputs.items():
            if model.tasks[name].kind == CLASSIFICATION:
                value = torch.softmax(out.float(), dim=-1)
            else:
                value = targets.scales[name].decode(out.float())
            chunks[name].append(value.cpu().numpy())
    return {name: np.concatenate(parts) for name, parts in chunks.items()}


class Predictor:
    """A trained model + tokenizer + target scaling, ready for inference."""

    def __init__(
        self, model: HRGPT, tokenizer, targets: TargetEncoder, max_tokens: int, info: dict
    ):
        self.model = model.eval()
        self.tokenizer = tokenizer
        self.targets = targets
        self.max_tokens = max_tokens
        self.info = info  # epoch, val metrics, ... from the checkpoint

    @property
    def tasks(self) -> dict[str, TaskSpec]:
        return self.model.tasks

    @classmethod
    def from_checkpoint(cls, path: str | Path, device: str = "auto") -> "Predictor":
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(
                f"checkpoint not found: {path}. Train one first with `hr-train` (see README)."
            )
        ckpt = torch.load(path, map_location="cpu", weights_only=True)
        tasks = tasks_from_dict(ckpt["tasks"])
        model = HRGPT(ModelConfig(**ckpt["model_config"]), tasks)
        model.load_state_dict(ckpt["model"])
        model.to(pick_device(device))
        tokenizer_dir = path.parent / ckpt["tokenizer_dir"]
        tokenizer = build_tokenizer(str(tokenizer_dir) if tokenizer_dir.exists() else "gpt2")
        targets = TargetEncoder.from_state_dict(tasks, ckpt["targets"])
        info = {k: ckpt.get(k) for k in ("epoch", "step", "val_loss", "val_metrics")}
        return cls(model, tokenizer, targets, ckpt["max_tokens"], info)

    def predict_many(self, records: list[dict[str, Any]]) -> dict[str, np.ndarray]:
        ids = [encode_text(self.tokenizer, record_to_text(r), self.max_tokens) for r in records]
        return run_model(self.model, ids, self.tokenizer.pad_token_id, self.targets)

    def predict(self, record: dict[str, Any]) -> dict[str, dict[str, Any]]:
        """One employee -> readable predictions for every task.

        Classification: {"label": "Leaves", "probabilities": {"Stays": 0.7, "Leaves": 0.3}}
        Regression:     {"value": 6123.4}
        """
        record = validate_record(record)
        raw = self.predict_many([record])
        result: dict[str, dict[str, Any]] = {}
        for name, spec in self.tasks.items():
            if spec.kind == CLASSIFICATION:
                probs = raw[name][0]
                result[name] = {
                    "label": spec.classes[int(probs.argmax())],
                    "probabilities": {c: float(p) for c, p in zip(spec.classes, probs)},
                }
            else:
                result[name] = {"value": float(raw[name][0])}
        return result


def actual_values(record: dict[str, Any], tasks: dict[str, TaskSpec]) -> dict[str, Any]:
    """The true targets of a dataset row, written like the predictions (for comparison)."""
    out = {}
    for name, spec in tasks.items():
        value = record.get(spec.column)
        if value is None:
            continue
        out[name] = spec.classes[spec.values.index(value)] if spec.kind == CLASSIFICATION else value
    return out


def format_prediction(
    result: dict[str, dict[str, Any]], actual: dict[str, Any] | None = None
) -> str:
    lines = []
    for name, res in result.items():
        if "label" in res:
            conf = res["probabilities"][res["label"]]
            text = f"{name:16s} {res['label']} ({conf:.0%})"
        else:
            text = f"{name:16s} ${res['value']:,.0f}"
        if actual and name in actual:
            truth = actual[name]
            text += f"   | actual: {truth if isinstance(truth, str) else f'${truth:,.0f}'}"
        lines.append(text)
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Predict attrition, job level, satisfaction and income."
    )
    parser.add_argument("--checkpoint", default="checkpoints/run/best.pt")
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--json", help="JSON file with one employee (the 27 input columns)")
    source.add_argument(
        "--test-row", type=int, help="use row N of the test split (shows the truth too)"
    )
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--device", default="auto")
    parser.add_argument(
        "--show-text", action="store_true", help="print the sentence the model reads"
    )
    args = parser.parse_args(argv)

    predictor = Predictor.from_checkpoint(args.checkpoint, args.device)
    actual = None
    if args.json:
        record = json.loads(Path(args.json).read_text())
    else:
        rows = frame_to_records(load_split(DataConfig(data_dir=Path(args.data_dir)), "test"))
        record = rows[args.test_row]
        actual = actual_values(record, predictor.tasks)
        record = {k: v for k, v in record.items() if k not in TARGET_COLUMNS}
    if args.show_text:
        print(record_to_text(record), "\n")
    print(format_prediction(predictor.predict(record), actual))


if __name__ == "__main__":
    main()
