"""Training end to end on synthetic data, then reload the checkpoint for inference and evaluation."""

import itertools
import json

import pytest
import torch

from hr_llm.config import TASKS, TrainConfig
from hr_llm.data import frame_to_records, load_split
from hr_llm.evaluate import evaluate_checkpoint, to_markdown
from hr_llm.features import TARGET_COLUMNS
from hr_llm.model import HRLLM
from hr_llm.predict import Predictor, actual_values, format_prediction
from hr_llm.train import lr_lambda, train


def test_lr_schedule_warms_up_then_decays_to_zero():
    values = [lr_lambda(s, warmup_steps=4, total_steps=20) for s in range(21)]
    assert values[0] == pytest.approx(0.25) and values[3] == pytest.approx(1.0)
    assert all(a >= b for a, b in itertools.pairwise(values[3:]))
    assert values[20] == pytest.approx(0.0, abs=1e-9)


def test_invalid_train_config_is_rejected():
    with pytest.raises(ValueError, match="precision"):
        TrainConfig(precision="fp8").validate()
    with pytest.raises(ValueError, match="epochs"):
        TrainConfig(epochs=0).validate()


@pytest.fixture()
def trained(tmp_path, data_config, tokenizer, tiny_model_config):
    torch.manual_seed(0)
    config = TrainConfig(
        output_dir=tmp_path / "run",
        epochs=3,
        batch_size=16,
        lr=3e-3,
        head_lr=3e-3,
        patience=0,
        log_every=1000,
        device="cpu",
    )
    model = HRLLM(tiny_model_config, TASKS)
    summary = train(data_config, tiny_model_config, config, tokenizer=tokenizer, model=model)
    return config.output_dir, summary


def test_training_writes_checkpoints_and_history(trained):
    out, summary = trained
    assert (out / "best.pt").exists() and (out / "last.pt").exists()
    assert (out / "tokenizer" / "tokenizer.json").exists()
    history = json.loads((out / "history.json").read_text())
    assert [h["epoch"] for h in history] == [1, 2, 3]
    assert history[-1]["train_loss"] < history[0]["train_loss"]
    assert summary["best_epoch"] in (1, 2, 3)
    assert set(history[0]["val"]) == set(TASKS)


def test_embeddings_are_frozen_by_default(trained, tiny_model_config):
    out, _ = trained
    ckpt = torch.load(out / "best.pt", weights_only=True)
    torch.manual_seed(0)
    fresh = HRLLM(tiny_model_config, TASKS)
    torch.testing.assert_close(
        ckpt["model"]["transformer.wte.weight"], fresh.state_dict()["transformer.wte.weight"]
    )


def test_checkpoint_reloads_for_prediction(trained, data_config):
    out, _ = trained
    predictor = Predictor.from_checkpoint(out / "best.pt", device="cpu")
    row = frame_to_records(load_split(data_config, "test"))[0]
    inputs = {k: v for k, v in row.items() if k not in TARGET_COLUMNS}
    result = predictor.predict(inputs)
    assert set(result) == set(TASKS)
    assert result["Attrition"]["label"] in ("Stays", "Leaves")
    assert sum(result["JobLevel"]["probabilities"].values()) == pytest.approx(1.0, abs=1e-5)
    assert result["MonthlyIncome"]["value"] > 0
    text = format_prediction(result, actual_values(row, predictor.tasks))
    assert "actual" in text and "$" in text
    # same answer twice (eval mode, no dropout)
    assert predictor.predict(inputs) == result


def test_missing_checkpoint_has_a_helpful_error(tmp_path):
    with pytest.raises(FileNotFoundError, match="hr-train"):
        Predictor.from_checkpoint(tmp_path / "nope.pt")


def test_evaluation_compares_with_baselines(trained, data_config):
    out, _ = trained
    results = evaluate_checkpoint(out / "best.pt", data_config)
    for name in TASKS:
        assert set(results[name]) == {"HR-LLM", "linear", "trivial"}
    assert results["Attrition"]["trivial"]["roc_auc"] == pytest.approx(0.5)
    table = to_markdown(results)
    assert "| ROC-AUC |" in table and "| R² |" in table
