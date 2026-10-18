"""Data pipeline: preparation, split, targets, tokens, batches."""

import math

import numpy as np
import pandas as pd
import pytest
import torch

from hr_llm.config import TASKS
from hr_llm.data import (
    Collator,
    HRDataset,
    TargetEncoder,
    clean_frame,
    download_csv,
    encode_text,
    load_split,
    prepare_data,
    stratified_split,
)
from hr_llm.features import ALL_COLUMNS, DROPPED_COLUMNS

from .conftest import synthetic_rows


def test_prepare_writes_three_disjoint_splits(data_config):
    paths = prepare_data(data_config)
    frames = {name: pd.read_csv(p) for name, p in paths.items()}
    assert sum(len(f) for f in frames.values()) == 120
    assert list(frames["train"].columns) == ALL_COLUMNS
    assert not set(DROPPED_COLUMNS) & set(frames["train"].columns)
    keys = {name: set(map(tuple, f.astype(str).values)) for name, f in frames.items()}
    assert not keys["train"] & keys["test"] and not keys["val"] & keys["test"]


def test_split_is_reproducible_and_stratified():
    df = pd.DataFrame(synthetic_rows(400))
    a = stratified_split(df, "Attrition", 0.1, 0.1, seed=3)
    b = stratified_split(df, "Attrition", 0.1, 0.1, seed=3)
    assert all(a[k].equals(b[k]) for k in a)
    rate = (df["Attrition"] == "Yes").mean()
    for frame in a.values():
        assert abs((frame["Attrition"] == "Yes").mean() - rate) < 0.03


def test_checksum_mismatch_is_reported(data_config):
    data_config.csv_md5 = "0" * 32
    with pytest.raises(ValueError, match="MD5"):
        download_csv(data_config)


def test_clean_frame_rejects_missing_columns():
    df = pd.DataFrame(synthetic_rows(3)).drop(columns=["Age"])
    with pytest.raises(ValueError, match="missing columns"):
        clean_frame(df)


def test_clean_frame_handles_bom_and_spaces():
    df = pd.DataFrame(synthetic_rows(3)).rename(columns={"Age": "﻿Age"})
    df["Gender"] = " " + df["Gender"] + " "
    out = clean_frame(df)
    assert "Age" in out.columns and set(out["Gender"]) <= {"Male", "Female"}


def test_targets_encode_and_decode(data_config):
    train = load_split(data_config, "train")
    enc = TargetEncoder.fit(TASKS, train)
    row = train.iloc[0].to_dict()
    t = enc.encode(row)
    assert t["Attrition"] == (1 if row["Attrition"] == "Yes" else 0)
    assert t["JobLevel"] == row["JobLevel"] - 1  # 1..5 -> 0..4
    assert t["JobSatisfaction"] == row["JobSatisfaction"] - 1
    back = enc.scales["MonthlyIncome"].decode(torch.tensor(t["MonthlyIncome"]))
    assert math.isclose(back.item(), row["MonthlyIncome"], rel_tol=1e-4)
    # standardised on the train split: mean ~0, std ~1
    z = np.array([enc.encode(r)["MonthlyIncome"] for r in train.to_dict(orient="records")])
    assert abs(z.mean()) < 1e-6 and abs(z.std() - 1) < 1e-6


def test_target_encoder_survives_a_checkpoint(data_config):
    train = load_split(data_config, "train")
    enc = TargetEncoder.fit(TASKS, train)
    again = TargetEncoder.from_state_dict(TASKS, enc.state_dict())
    row = train.iloc[3].to_dict()
    assert again.encode(row) == enc.encode(row)


def test_unknown_target_value_is_an_error(data_config):
    enc = TargetEncoder.fit(TASKS, load_split(data_config, "train"))
    row = {**synthetic_rows(1)[0], "Attrition": "Maybe"}
    with pytest.raises(ValueError, match="Attrition"):
        enc.encode(row)


def test_balanced_class_weights_favour_the_rare_class(data_config):
    train = load_split(data_config, "train")
    enc = TargetEncoder.fit(TASKS, train)
    w = enc.class_weights("Attrition", train)
    assert w is not None and w[1] > w[0]
    assert enc.class_weights("JobLevel", train) is None  # not a balanced task


def test_encode_text_ends_with_eos_and_respects_max_tokens(tokenizer):
    ids = encode_text(tokenizer, "The employee is 41 years old.", max_tokens=4)
    assert len(ids) == 4 and ids[-1] == tokenizer.eos_token_id


def test_dataset_and_collator(data_config, tokenizer):
    train = load_split(data_config, "train")
    ds = HRDataset(train, tokenizer, TargetEncoder.fit(TASKS, train), max_tokens=512)
    assert len(ds) == len(train)
    assert tokenizer.unk_token_id not in ds.ids[0], "test tokenizer should know every word"
    items = [ds[0], ds[1], ds[2]]
    batch = Collator(tokenizer.pad_token_id)(items)
    lengths = [len(it["input_ids"]) for it in items]
    assert batch["lengths"].tolist() == lengths
    assert batch["input_ids"].shape == (3, max(lengths))
    for row, n in enumerate(lengths):
        assert batch["input_ids"][row, n - 1] == tokenizer.eos_token_id
        assert (batch["input_ids"][row, n:] == tokenizer.pad_token_id).all()
    assert batch["targets"]["Attrition"].dtype == torch.long
    assert batch["targets"]["MonthlyIncome"].dtype == torch.float32
