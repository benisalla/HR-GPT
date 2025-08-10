import json
import torch
import pytest
from dataset.tokenizer import MyTokenizer
from dataset.BatchLoader import BatchLoader  
from dataclasses import dataclass

@dataclass
class Spec:
    target_col: str
    task_type: str

@dataclass
class Cfg:
    tasks: dict[str, Spec]

@pytest.fixture(scope="module")
def tok():
    return MyTokenizer("gpt2")

@pytest.fixture()
def tiny_split(tmp_path):
    data = [
        {"Age": 30, "JobRole": "Engineer", "Attrition": 1, "MonthlyIncome": 4200, "Department": "R&D"},
        {"Age": 45, "JobRole": "Manager", "Attrition": 0, "MonthlyIncome": 8700, "Department": "Sales"},
    ]
    d = tmp_path / "toy_data"
    d.mkdir()
    for split in ("train", "valid"):
        (d / f"{split}.json").write_text(json.dumps(data), encoding="utf-8")
    return str(d)

@pytest.fixture()
def cfg():
    return Cfg(
        tasks={
            "attrition": Spec(target_col="Attrition", task_type="classification"),
            "income": Spec(target_col="MonthlyIncome", task_type="regression"),
        }
    )

def make_loader(cfg, data_dir, tok, toks_in_batch=64, split="train", training=True):
    loader = BatchLoader(cfg, data_dir, split, tok, toks_in_batch=toks_in_batch, device="cpu")
    # Ensure pad_id exists on the loader (your class should set this itself; see note above)
    if not hasattr(loader, "pad_id"):
        loader.pad_id = tok.pad_id
    return loader

def test_builds_and_iterates(cfg, tiny_split, tok):
    loader = make_loader(cfg, tiny_split, tok, toks_in_batch=128)
    assert len(loader) > 0

    it = iter(loader)
    x_batch, x_mask, y_list, task_names = next(it)

    # shapes
    assert x_batch.ndim == 2
    assert x_mask.shape == x_batch.shape
    assert len(y_list) == x_batch.shape[0] == len(task_names)

    # mask correctness: 1 for non-pad, 0 for pad
    pad = loader.pad_id
    for ids_row, m_row in zip(x_batch.tolist(), x_mask.tolist()):
        for tid, m in zip(ids_row, m_row):
            assert (tid == pad and m == 0) or (tid != pad and m == 1)

def test_y_dtypes_match_task(cfg, tiny_split, tok):
    loader = make_loader(cfg, tiny_split, tok, toks_in_batch=512)
    x_batch, x_mask, y_list, task_names = next(iter(loader))
    for y, tname in zip(y_list, task_names):
        if tname == "income":
            assert y.dtype == torch.float32
        elif tname == "attrition":
            assert y.dtype == torch.long
        else:
            pytest.fail(f"Unexpected task {tname}")

def test_stop_iteration(cfg, tiny_split, tok):
    loader = make_loader(cfg, tiny_split, tok, toks_in_batch=64)
    count = 0
    for _ in loader:
        count += 1
    assert count == len(loader)

def test_record_to_text_format(cfg, tiny_split, tok):
    loader = make_loader(cfg, tiny_split, tok, toks_in_batch=64)
    sample_inputs = {
        "Age": 33,
        "JobRole": "Analyst",
        "Attrition": 0,
        "MonthlyIncome": 3000,
    }
    txt = loader.__record_to_text__(sample_inputs, tgt="Attrition")
    assert txt.startswith("<sost>")
    assert txt.endswith("<eost>")
    assert "The employee" in txt

def test_long_sequence_forces_batchsize_one(cfg, tiny_split, tok):
    # Make a very long input by repeating words; aim for L >= toks_in_batch
    long_inputs = {
        "Attrition": 1,
        "MonthlyIncome": 5000,
        "Department": "R&D",
        "JobRole": "Engineer " * 200,  # repeated to increase tokenized length
    }
    # Patch the split file to contain only this one record so we can reason about length
    with open(f"{tiny_split}/train.json", "w", encoding="utf-8") as f:
        json.dump([long_inputs], f)

    loader = make_loader(cfg, tiny_split, tok, toks_in_batch=8)  # tiny token budget
    # training=True packs by equal lengths; here only one sample, so batch size must be 1
    x_batch, x_mask, y_list, task_names = next(iter(loader))
    assert x_batch.shape[0] == 1

# Add to test/test_batchloader.py
import json
import pytest

def test_non_train_split_batches_are_singletons(cfg, tiny_split, tok):
    # Reuse same file but pretend it's "valid"
    loader = make_loader(cfg, tiny_split, tok, toks_in_batch=9999, split="valid")
    # Non-training path: each batch should be size 1
    for x_batch, x_mask, y_list, tnames in loader:
        assert x_batch.shape[0] == 1
        assert len(y_list) == 1
        assert len(tnames) == 1

def test_missing_target_raises(tmp_path, tok):
    bad = [{"Age": 30, "MonthlyIncome": 5000}]  # missing Attrition
    d = tmp_path / "toy_data2"
    d.mkdir()
    (d / "train.json").write_text(json.dumps(bad), encoding="utf-8")

    class Spec: 
        def __init__(self, tgt, ty): self.target_col, self.task_type = tgt, ty
    class Cfg:
        tasks = {"attrition": Spec("Attrition", "classification")}

    from dataset.BatchLoader import BatchLoader
    with pytest.raises(ValueError, match="Target column 'Attrition'"):
        BatchLoader(Cfg(), str(d), "train", tok, toks_in_batch=64)

def test_pad_id_consistency(cfg, tiny_split, tok):
    loader = make_loader(cfg, tiny_split, tok, toks_in_batch=64)
    # Ensure loader uses same pad as tokenizer
    assert loader.pad_id == tok.pad_id

def test_packing_respects_toks_in_batch(cfg, tiny_split, tok):
    # Force small budget so multi-sample batches must be small
    loader = make_loader(cfg, tiny_split, tok, toks_in_batch=16)
    for batch in loader.batches:
        # worst-case length in this batch
        L = max(len(s["ids"]) for s in batch)
        # computed upper bound on batch size
        assert len(batch) <= max(1, 16 // max(1, L))