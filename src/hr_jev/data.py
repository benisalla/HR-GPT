"""Data pipeline: download the CSV, split it, turn each employee into tokens and targets.

raw CSV --prepare_data--> data/processed/{train,val,test}.csv
row --record_to_text--> "The employee is 41 years old, ..." --tokenizer--> ids + <eos>
row --TargetEncoder--> {"Attrition": 1, "JobLevel": 1, ..., "MonthlyIncome": 0.37}
"""

import hashlib
import json
import math
import urllib.request
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from hr_jev.config import CLASSIFICATION, DataConfig, TaskSpec
from hr_jev.features import (
    ALL_COLUMNS,
    DROPPED_COLUMNS,
    FEATURES,
    TARGET_COLUMNS,
    record_to_text,
)

PAD_TOKEN = "<|pad|>"
SPLITS = ("train", "val", "test")


# ---------------------------------------------------------------------------
# Download and split


def download_csv(config: DataConfig) -> Path:
    """Download the IBM HR Analytics CSV once (1470 rows, ~230 KB) and check its MD5."""
    path = config.raw_csv
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        print(f"downloading {config.csv_url}")
        tmp = path.with_suffix(".part")
        urllib.request.urlretrieve(config.csv_url, tmp)
        tmp.rename(path)
    digest = hashlib.md5(path.read_bytes()).hexdigest()
    if config.csv_md5 and digest != config.csv_md5:
        raise ValueError(f"{path} has MD5 {digest}, expected {config.csv_md5}; delete it and retry")
    return path


def clean_frame(df: pd.DataFrame) -> pd.DataFrame:
    """Keep the known columns, strip text, check there are no missing values."""
    df = df.rename(columns=lambda c: c.strip().lstrip("﻿"))
    missing = [c for c in ALL_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"CSV is missing columns: {missing}")
    df = df.drop(columns=[c for c in DROPPED_COLUMNS if c in df.columns])[ALL_COLUMNS].copy()
    for col in df.columns:
        if not pd.api.types.is_numeric_dtype(df[col]):
            df[col] = df[col].astype(str).str.strip()
    if df.isna().any().any():
        raise ValueError(f"missing values in: {df.columns[df.isna().any()].tolist()}")
    return df


def stratified_split(
    df: pd.DataFrame, column: str, val_fraction: float, test_fraction: float, seed: int
) -> dict[str, pd.DataFrame]:
    """Shuffle and split so each split keeps the same share of each `column` value.

    Attrition is rare (16%), so a plain random split can leave val/test with too
    few leavers to measure anything.
    """
    rng = np.random.default_rng(seed)
    parts: dict[str, list[np.ndarray]] = {name: [] for name in SPLITS}
    for _, group in df.groupby(column, sort=True):
        idx = rng.permutation(group.index.to_numpy())
        n_val = round(len(idx) * val_fraction)
        n_test = round(len(idx) * test_fraction)
        parts["test"].append(idx[:n_test])
        parts["val"].append(idx[n_test : n_test + n_val])
        parts["train"].append(idx[n_test + n_val :])
    return {
        name: df.loc[rng.permutation(np.concatenate(chunks))].reset_index(drop=True)
        for name, chunks in parts.items()
    }


def prepare_data(config: DataConfig, force: bool = False) -> dict[str, Path]:
    """Download (if needed), clean and split the data. Returns the path of each split."""
    out = config.processed_dir
    paths = {name: out / f"{name}.csv" for name in SPLITS}
    if not force and all(p.exists() for p in paths.values()):
        return paths
    df = clean_frame(pd.read_csv(download_csv(config)))
    splits = stratified_split(
        df, "Attrition", config.val_fraction, config.test_fraction, config.split_seed
    )
    out.mkdir(parents=True, exist_ok=True)
    stats = {}
    for name, frame in splits.items():
        frame.to_csv(paths[name], index=False)
        stats[name] = {
            "rows": len(frame),
            "attrition_rate": round(float((frame["Attrition"] == "Yes").mean()), 4),
        }
    (out / "split_stats.json").write_text(json.dumps(stats, indent=2))
    print("prepared:", ", ".join(f"{n}={s['rows']} rows" for n, s in stats.items()))
    return paths


def load_split(config: DataConfig, split: str) -> pd.DataFrame:
    if split not in SPLITS:
        raise ValueError(f"split must be one of {SPLITS}, got {split!r}")
    return pd.read_csv(prepare_data(config)[split])


def frame_to_records(df: pd.DataFrame) -> list[dict[str, Any]]:
    """DataFrame -> list of dicts with plain Python values (numpy ints become int)."""
    return [
        {k: (v.item() if hasattr(v, "item") else v) for k, v in row.items()}
        for row in df.to_dict(orient="records")
    ]


# ---------------------------------------------------------------------------
# Targets


@dataclass
class RegressionScale:
    """Standardise a regression target: z = (f(y) - mean) / std, f = log or identity."""

    mean: float
    std: float
    log: bool

    def encode(self, value: float) -> float:
        x = math.log(value) if self.log else float(value)
        return (x - self.mean) / self.std

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        x = z * self.std + self.mean
        return torch.exp(x) if self.log else x


class TargetEncoder:
    """Turns raw CSV targets into training labels, and model outputs back into values.

    Regression scales are fitted on the training split only, then stored in the
    checkpoint so inference uses exactly the same numbers.
    """

    def __init__(self, tasks: dict[str, TaskSpec], scales: dict[str, RegressionScale]):
        self.tasks = tasks
        self.scales = scales

    @classmethod
    def fit(cls, tasks: dict[str, TaskSpec], train: pd.DataFrame) -> "TargetEncoder":
        scales = {}
        for name, spec in tasks.items():
            if spec.kind != CLASSIFICATION:
                values = train[spec.column].astype(float).to_numpy()
                x = np.log(values) if spec.log_target else values
                scales[name] = RegressionScale(float(x.mean()), float(x.std()), spec.log_target)
        return cls(tasks, scales)

    def encode(self, record: dict[str, Any]) -> dict[str, float | int]:
        targets: dict[str, float | int] = {}
        for name, spec in self.tasks.items():
            value = record[spec.column]
            if spec.kind == CLASSIFICATION:
                if value not in spec.values:
                    raise ValueError(f"{name}: unknown value {value!r}, expected {spec.values}")
                targets[name] = spec.values.index(value)
            else:
                targets[name] = self.scales[name].encode(value)
        return targets

    def class_weights(self, name: str, train: pd.DataFrame) -> torch.Tensor | None:
        """1 / frequency weights (mean 1) for a `balanced` task, else None."""
        spec = self.tasks[name]
        if spec.kind != CLASSIFICATION or not spec.balanced:
            return None
        counts = np.array([(train[spec.column] == v).sum() for v in spec.values], dtype=float)
        weights = counts.sum() / (len(counts) * np.maximum(counts, 1))
        return torch.tensor(weights, dtype=torch.float32)

    def state_dict(self) -> dict[str, Any]:
        return {name: vars(scale) for name, scale in self.scales.items()}

    @classmethod
    def from_state_dict(cls, tasks: dict[str, TaskSpec], state: dict[str, Any]) -> "TargetEncoder":
        return cls(tasks, {name: RegressionScale(**s) for name, s in state.items()})


# ---------------------------------------------------------------------------
# Tokenizer


def build_tokenizer(name_or_path: str = "gpt2"):
    """GPT-2 BPE tokenizer with an extra <|pad|> token (id 50257)."""
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(name_or_path)
    if tokenizer.pad_token is None:
        tokenizer.add_special_tokens({"pad_token": PAD_TOKEN})
    return tokenizer


def encode_text(tokenizer, text: str, max_tokens: int) -> list[int]:
    """text -> token ids + <eos>. The model reads the <eos> position as the summary."""
    ids = tokenizer.encode(text, add_special_tokens=False)[: max_tokens - 1]
    return ids + [tokenizer.eos_token_id]


# ---------------------------------------------------------------------------
# Dataset and batching


class HRDataset(Dataset):
    """One item per employee: token ids of its sentence + its targets for every task."""

    def __init__(
        self,
        frame: pd.DataFrame,
        tokenizer,
        targets: TargetEncoder,
        max_tokens: int,
    ):
        self.records = frame_to_records(frame)
        self.texts = [record_to_text(r) for r in self.records]
        self.ids = [encode_text(tokenizer, t, max_tokens) for t in self.texts]
        self.targets = [targets.encode(r) for r in self.records]

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> dict[str, Any]:
        return {"input_ids": self.ids[index], "targets": self.targets[index]}


@dataclass
class Collator:
    """Pad a list of items to the longest sentence; padding goes at the end.

    The model is causal, so padding after the last real token cannot change it,
    and `lengths` says where that last token is.
    """

    pad_token_id: int

    def __call__(self, items: Sequence[dict[str, Any]]) -> dict[str, Any]:
        lengths = torch.tensor([len(it["input_ids"]) for it in items])
        input_ids = torch.full((len(items), int(lengths.max())), self.pad_token_id)
        for row, item in enumerate(items):
            input_ids[row, : len(item["input_ids"])] = torch.tensor(item["input_ids"])
        targets = {
            name: torch.tensor(
                [it["targets"][name] for it in items],
                dtype=torch.long if isinstance(items[0]["targets"][name], int) else torch.float32,
            )
            for name in items[0]["targets"]
        }
        return {"input_ids": input_ids, "lengths": lengths, "targets": targets}


def describe_columns() -> str:
    """Short summary of inputs and targets, for logs and docs."""
    return f"{len(FEATURES)} input columns -> {len(TARGET_COLUMNS)} targets ({TARGET_COLUMNS})"
