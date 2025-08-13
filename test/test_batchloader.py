import pytest
import torch
import json
from dataclasses import replace
from pathlib import Path
import dataset.BatchLoader as BLmod
from dataset.tokenizer import MyTokenizer
from dataset.BatchLoader import BatchLoader
from model.config import GPTConfig, TaskSpec, DatasetConfig


@pytest.fixture(scope="module")
def tok():
    return MyTokenizer()

@pytest.fixture()
def tiny_split(tmp_path: Path):
    data = [
        {"Age": 30, "JobRole": "Engineer", "Attrition": 1, "MonthlyIncome": 4200, "Department": "R&D"},
        {"Age": 45, "JobRole": "Manager",  "Attrition": 0, "MonthlyIncome": 8700, "Department": "Sales"},
        {"Age": 27, "JobRole": "Analyst",  "Attrition": 1, "MonthlyIncome": 3100, "Department": "R&D"},
    ]
    d = tmp_path / "toy_data"
    d.mkdir()
    for split in ("train", "valid"):
        (d / f"{split}.json").write_text(json.dumps(data), encoding="utf-8")
    return str(d)

@pytest.fixture()
def cfg(tiny_split):
    tasks = {
        "Attrition": TaskSpec(target_col="Attrition", task_type="binary", n_classes=2, mult_fact=2, dropout=0.5),
        "MonthlyIncome": TaskSpec(target_col="MonthlyIncome", task_type="regression", mult_fact=4, dropout=0.5),
    }
    dataset = DatasetConfig(data_dir=tiny_split, toks_in_batch=64, batch_size=8)
    return GPTConfig(tasks=tasks, dataset=dataset).validate()

@pytest.fixture(autouse=True)
def patch_utils(monkeypatch):
    """
    Make tests independent from dataset.utils by stubbing ORDERED_ATTRS and _pretty_attr
    inside the dataset.BatchLoader module (import site).
    """
    monkeypatch.setattr(
        BLmod, "ORDERED_ATTRS",
        ["Age", "Department", "JobRole", "Attrition", "MonthlyIncome"],
        raising=False,
    )
    def _pretty_attr_stub(attr, val):
        return f"{attr} is {val}"
    monkeypatch.setattr(BLmod, "_pretty_attr", _pretty_attr_stub, raising=False)

def make_loader(cfg: GPTConfig, tok: MyTokenizer, split="train", device="cpu"):
    return BatchLoader(cfg, split, tok, device=device)

def test_builds_and_iterates(cfg, tok):
    loader = make_loader(cfg, tok, split="train")
    assert len(loader) > 0

    x_batch, x_mask, y_list, task_names = next(iter(loader))

    # shapes & alignment
    assert x_batch.ndim == 2
    assert x_mask.shape == x_batch.shape
    assert len(y_list) == x_batch.shape[0] == len(task_names)

    # mask correctness: 1 for non-pad, 0 for pad
    pad = loader.pad_id
    for ids_row, m_row in zip(x_batch.tolist(), x_mask.tolist()):
        for tid, m in zip(ids_row, m_row):
            assert (tid == pad and m == 0) or (tid != pad and m == 1)

def test_y_types_and_values(cfg, tok):
    loader = make_loader(cfg, tok, split="train")
    x_batch, x_mask, y_list, tnames = next(iter(loader))
    for y, tname in zip(y_list, tnames):
        if tname == "MonthlyIncome":
            assert y.dtype == torch.float32
            assert float(y) in {4200.0, 8700.0, 3100.0}
        elif tname == "Attrition":
            assert y.dtype == torch.long
            assert int(y) in {0, 1}
        else:
            pytest.fail(f"Unexpected task {tname}")

def test_non_train_split_batches_are_singletons(cfg, tok):
    loader = make_loader(cfg, tok, split="valid")
    for x_batch, x_mask, y_list, tnames in loader:
        assert x_batch.shape[0] == 1
        assert len(y_list) == 1
        assert len(tnames) == 1

def test_stop_iteration(cfg, tok):
    loader = make_loader(cfg, tok, split="train")
    count = sum(1 for _ in loader)
    assert count == len(loader)

def test_pad_id_consistency(cfg, tok):
    loader = make_loader(cfg, tok, split="train")
    assert loader.pad_id == tok.pad_id

def test_samples_are_sorted_by_length_when_training(cfg, tok):
    loader = make_loader(cfg, tok, split="train")
    lens = [s["len"] for s in loader.samples]
    assert lens == sorted(lens), "Training loader should sort samples by length"

def test_packing_respects_toks_in_batch(cfg, tok):
    # Clone cfg with a tiny token budget to hit the constraint reliably
    cfg_small = replace(cfg, dataset=replace(cfg.dataset, toks_in_batch=16))
    loader = make_loader(cfg_small, tok, split="train")
    for batch in loader.batches:
        L = max(len(s["ids"]) for s in batch)  # worst-case seq length in batch
        assert len(batch) <= max(1, cfg_small.dataset.toks_in_batch // max(1, L))

def test_long_sequence_forces_batch_size_one(cfg, tok, tiny_split):
    # Overwrite train.json with a single very long record
    long_rec = {
        "Age": 40,
        "Department": "R&D",
        "JobRole": "Engineer " * 500,  # very long
        "Attrition": 1,
        "MonthlyIncome": 6000,
    }
    train_file = Path(tiny_split) / "train.json"
    train_file.write_text(json.dumps([long_rec]), encoding="utf-8")

    cfg_tiny = replace(cfg, dataset=replace(cfg.dataset, toks_in_batch=8))
    loader = make_loader(cfg_tiny, tok, split="train")
    x_batch, x_mask, y_list, tnames = next(iter(loader))
    assert x_batch.shape[0] == 1

def test_record_to_text_excludes_target_and_formats(cfg, tok):
    loader = make_loader(cfg, tok, split="train")
    sample_inputs = {
        "Age": 33,
        "JobRole": "Analyst",
        "Attrition": 0,          # target for one task
        "MonthlyIncome": 3000,   # target for another task
        "Department": "Sales",
    }
    txt = loader.__record_to_text__(sample_inputs, tgt="Attrition")
    assert "Attrition" not in txt  # target excluded
    assert txt.startswith("The employee ")
    assert txt.endswith(".")

def test_missing_target_raises(tok, tmp_path):
    bad = [{"Age": 30, "MonthlyIncome": 5000}]  # missing Attrition
    d = Path(tmp_path) / "toy_bad"
    d.mkdir()
    (d / "train.json").write_text(json.dumps(bad), encoding="utf-8")

    tasks = {"Attrition": TaskSpec("Attrition", "binary", n_classes=2)}
    cfg_bad = GPTConfig(tasks=tasks, dataset=DatasetConfig(data_dir=str(d), toks_in_batch=32, batch_size=4)).validate()

    with pytest.raises(ValueError, match="Target column 'Attrition'"):
        BatchLoader(cfg_bad, "train", tok)

def test_attention_mask_matches_nonpad(cfg, tok):
    loader = make_loader(cfg, tok, split="train")
    x_batch, x_mask, *_ = next(iter(loader))
    truth = (x_batch != loader.pad_id).long()
    assert torch.equal(truth, x_mask)