"""Model: shapes, causality, padding, loss, optimizer groups, GPT-2 weight loading."""

import pytest
import torch
from transformers import GPT2Config, GPT2Model

from hr_llm import model as model_module
from hr_llm.config import TASKS, ModelConfig
from hr_llm.model import HRLLM, build_model, load_pretrained_backbone


@pytest.fixture()
def model(tiny_model_config):
    torch.manual_seed(0)
    return HRLLM(tiny_model_config, TASKS).eval()


def random_batch(vocab: int, lengths: list[int], pad: int):
    ids = torch.full((len(lengths), max(lengths)), pad)
    for i, n in enumerate(lengths):
        ids[i, :n] = torch.randint(1, vocab - 2, (n,))
    return ids, torch.tensor(lengths)


def test_output_shapes(model, tiny_model_config):
    ids, lengths = random_batch(tiny_model_config.vocab_size, [7, 5, 9], pad=0)
    out = model(ids, lengths)
    assert out["Attrition"].shape == (3, 2)
    assert out["JobLevel"].shape == (3, 5)
    assert out["JobSatisfaction"].shape == (3, 4)
    assert out["MonthlyIncome"].shape == (3,)


def test_padding_does_not_change_the_prediction(model, tiny_model_config):
    """The old model read position -1, which is a pad token for shorter rows."""
    ids, lengths = random_batch(tiny_model_config.vocab_size, [6, 10], pad=1)
    alone = model(ids[:1, :6], lengths[:1])
    batched = model(ids, lengths)
    for name in TASKS:
        torch.testing.assert_close(batched[name][:1], alone[name], atol=1e-5, rtol=1e-4)


def test_attention_is_causal(model):
    x = torch.randn(1, 8, model.config.n_embd)
    y = x.clone()
    y[:, 5:] += torch.randn(1, 3, model.config.n_embd)
    attn = model.transformer.h[0].attn
    torch.testing.assert_close(attn(x)[:, :5], attn(y)[:, :5])


def test_loss_is_finite_and_reaches_every_head(model, tiny_model_config):
    model.train()
    ids, lengths = random_batch(tiny_model_config.vocab_size, [5, 5, 5, 5], pad=0)
    targets = {
        "Attrition": torch.tensor([0, 1, 0, 1]),
        "JobLevel": torch.tensor([0, 1, 4, 2]),
        "JobSatisfaction": torch.tensor([3, 2, 1, 0]),
        "MonthlyIncome": torch.tensor([0.1, -1.0, 2.0, 0.3]),
    }
    loss, parts = model.loss(model(ids, lengths), targets, {"Attrition": torch.tensor([0.6, 3.0])})
    assert torch.isfinite(loss) and set(parts) == set(TASKS)
    loss.backward()
    for name in TASKS:
        assert all(p.grad is not None for p in model.heads[name].parameters())


def test_param_groups_cover_every_parameter_once(model):
    groups = model.param_groups(lr=1e-5, head_lr=1e-3, weight_decay=0.1)
    seen = [id(p) for g in groups for p in g["params"]]
    assert len(seen) == len(set(seen)) == len(list(model.parameters()))
    for g in groups:
        is_head = any(p is q for p in g["params"] for q in model.heads.parameters())
        assert g["lr"] == (1e-3 if is_head else 1e-5)
        if g["weight_decay"]:
            assert all(p.dim() >= 2 for p in g["params"])


def test_frozen_parameters_are_left_out_of_the_optimizer(model):
    model.transformer.wte.weight.requires_grad_(False)
    groups = model.param_groups(lr=1e-5, head_lr=1e-3, weight_decay=0.1)
    assert not any(p is model.transformer.wte.weight for g in groups for p in g["params"])


def test_can_overfit_a_small_batch(tiny_model_config):
    torch.manual_seed(0)
    m = HRLLM(tiny_model_config, TASKS)
    ids, lengths = random_batch(tiny_model_config.vocab_size, [6, 8, 7, 5], pad=0)
    targets = {
        "Attrition": torch.tensor([0, 1, 0, 1]),
        "JobLevel": torch.tensor([0, 1, 4, 2]),
        "JobSatisfaction": torch.tensor([3, 2, 1, 0]),
        "MonthlyIncome": torch.tensor([0.5, -1.0, 1.5, 0.0]),
    }
    opt = torch.optim.AdamW(m.parameters(), lr=3e-3)
    for _ in range(150):
        loss, _ = m.loss(m(ids, lengths), targets)
        opt.zero_grad()
        loss.backward()
        opt.step()
    assert loss.item() < 0.05


def test_too_long_input_is_rejected(model):
    ids = torch.zeros(1, model.config.block_size + 1, dtype=torch.long)
    with pytest.raises(ValueError, match="block_size"):
        model(ids, torch.tensor([ids.shape[1]]))


def test_invalid_config_is_rejected():
    with pytest.raises(ValueError, match="divisible"):
        ModelConfig(init="scratch", n_embd=30, n_head=4).validate()
    with pytest.raises(ValueError, match="init"):
        ModelConfig(init="gpt5").validate()


def test_gpt2_weights_give_the_same_hidden_states_as_hugging_face(monkeypatch):
    """Load a (tiny, random) Hugging Face GPT-2 into our backbone: outputs must match exactly.

    This proves the weight mapping (incl. the Conv1D transposes) and that our blocks
    compute the same function as GPT-2.
    """
    torch.manual_seed(0)
    hf_config = GPT2Config(
        vocab_size=100,
        n_positions=64,
        n_embd=32,
        n_layer=2,
        n_head=4,
        resid_pdrop=0.0,
        embd_pdrop=0.0,
        attn_pdrop=0.0,
    )
    hf = GPT2Model(hf_config).eval()
    monkeypatch.setattr(GPT2Model, "from_pretrained", classmethod(lambda cls, name: hf))
    ours_config = ModelConfig(
        init="gpt2",
        vocab_size=101,
        block_size=64,
        n_layer=2,
        n_head=4,
        n_embd=32,
        dropout=0.0,
        head_dropout=0.0,
    )
    ours = HRLLM(ours_config, TASKS).eval()
    copied = load_pretrained_backbone(ours, "gpt2")
    assert copied == sum(1 for k in ours.state_dict() if k.startswith("transformer."))

    ids = torch.randint(0, 100, (2, 12))
    expected = hf(ids).last_hidden_state[:, -1]
    got = ours.encode(ids, torch.tensor([12, 12]))
    torch.testing.assert_close(got, expected, atol=1e-5, rtol=1e-4)
    # the extra <|pad|> row is the mean of the real embeddings
    torch.testing.assert_close(ours.transformer.wte.weight[100], hf.wte.weight.mean(0))


def test_build_model_takes_its_size_from_the_pretrained_config(monkeypatch):
    monkeypatch.setattr(
        model_module,
        "config_for_pretrained",
        lambda name, base: ModelConfig(
            init=name, vocab_size=base.vocab_size, block_size=32, n_layer=1, n_head=2, n_embd=16
        ),
    )
    m = build_model(ModelConfig(init="distilgpt2"), TASKS, load_weights=False)
    assert m.config.n_layer == 1 and m.config.init == "distilgpt2"
