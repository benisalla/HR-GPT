"""Web app: the results table and the app callbacks, without a browser."""

import pytest

from hr_llm.app import results_table

PRED = {
    "Attrition": {"label": "Leaves", "probabilities": {"Stays": 0.3, "Leaves": 0.7}},
    "JobLevel": {"label": "Level 2", "probabilities": {"Level 1": 0.2, "Level 2": 0.8}},
    "MonthlyIncome": {"value": 5500.0},
}


def test_results_table_without_truth():
    table = results_table(PRED, None)
    assert "| Will they leave? | **Leaves** · 70% |" in table
    assert "**$5,500**" in table and "Actual" not in table


def test_results_table_with_truth_marks_hits_and_misses():
    table = results_table(
        PRED, {"Attrition": "Stays", "JobLevel": "Level 2", "MonthlyIncome": 5000}
    )
    assert "**Leaves** · 70% | Stays ❌ |" in table
    assert "**Level 2** · 80% | Level 2 ✅ |" in table
    assert "$5,000 (off by 10%)" in table


def test_app_builds_with_a_trained_checkpoint(tmp_path, data_config, tokenizer, tiny_model_config):
    gr = pytest.importorskip("gradio")
    from hr_llm.app import build_app
    from hr_llm.config import TASKS, TrainConfig
    from hr_llm.data import frame_to_records, load_split
    from hr_llm.model import HRLLM
    from hr_llm.predict import Predictor
    from hr_llm.train import train

    cfg = TrainConfig(output_dir=tmp_path / "run", epochs=1, log_every=1000, device="cpu")
    train(
        data_config,
        tiny_model_config,
        cfg,
        tokenizer=tokenizer,
        model=HRLLM(tiny_model_config, TASKS),
    )
    predictor = Predictor.from_checkpoint(tmp_path / "run" / "best.pt", device="cpu")
    demo = build_app(predictor, frame_to_records(load_split(data_config, "test")))
    assert isinstance(demo, gr.Blocks)
