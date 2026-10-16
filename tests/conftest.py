"""Shared fixtures: a synthetic HR CSV, a small offline tokenizer and a tiny model config.

Nothing here touches the network, so the whole suite runs offline on a CPU.
"""

import random
import re
from pathlib import Path

import pandas as pd
import pytest
from transformers import PreTrainedTokenizerFast

from hr_gpt.config import DataConfig, ModelConfig
from hr_gpt.features import DROPPED_COLUMNS, FEATURES, record_to_text

EOS, PAD, UNK = "<|endoftext|>", "<|pad|>", "<unk>"


def synthetic_rows(n: int, seed: int = 0) -> list[dict]:
    """Valid random employees, with targets that depend a little on the inputs."""
    rng = random.Random(seed)
    rows = []
    for i in range(n):
        row = {}
        for f in FEATURES:
            if f.is_categorical:
                row[f.name] = rng.choice(list(f.choices))
            else:
                row[f.name] = rng.randint(int(f.minimum), int(f.maximum))
        level = min(5, 1 + row["TotalWorkingYears"] // 9)
        row["Attrition"] = (
            "Yes" if (row["OverTime"] == "Yes" and i % 3 == 0) or i % 7 == 0 else "No"
        )
        row["JobLevel"] = level
        row["JobSatisfaction"] = rng.randint(1, 4)
        row["MonthlyIncome"] = 1500 * level + rng.randint(0, 1500)
        row.update(
            {"EmployeeCount": 1, "EmployeeNumber": i + 1, "Over18": "Y", "StandardHours": 80}
        )
        rows.append(row)
    return rows


@pytest.fixture()
def data_config(tmp_path: Path) -> DataConfig:
    """A data dir whose raw CSV is already in place (no download, no checksum)."""
    config = DataConfig(data_dir=tmp_path / "data", csv_md5="")
    config.raw_csv.parent.mkdir(parents=True)
    columns = list(synthetic_rows(1)[0])
    assert set(DROPPED_COLUMNS) <= set(columns)
    pd.DataFrame(synthetic_rows(120), columns=columns).to_csv(config.raw_csv, index=False)
    return config


def build_test_tokenizer() -> PreTrainedTokenizerFast:
    """Word-level tokenizer over the words of the feature sentences; anything else is <unk>."""
    from tokenizers import Tokenizer, models, pre_tokenizers

    words = set()
    for seed in range(5):
        for row in synthetic_rows(40, seed):
            words.update(re.findall(r"\w+|[^\w\s]+", record_to_text(row)))
    vocab = {tok: i for i, tok in enumerate([UNK, *sorted(words), EOS])}
    tk = Tokenizer(models.WordLevel(vocab, unk_token=UNK))
    tk.pre_tokenizer = pre_tokenizers.Whitespace()
    tokenizer = PreTrainedTokenizerFast(tokenizer_object=tk, unk_token=UNK, eos_token=EOS)
    tokenizer.add_special_tokens({"pad_token": PAD})
    return tokenizer


@pytest.fixture(scope="session")
def tokenizer() -> PreTrainedTokenizerFast:
    return build_test_tokenizer()


@pytest.fixture()
def tiny_model_config(tokenizer) -> ModelConfig:
    return ModelConfig(
        init="scratch",
        vocab_size=len(tokenizer),
        block_size=256,
        n_layer=2,
        n_head=2,
        n_embd=32,
        dropout=0.0,
        head_dropout=0.0,
    )
