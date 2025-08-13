from pathlib import Path
import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F
from model.config import GPTConfig, TrainConfig
from model.model import HRGPT, ClassificationHead, RegressionHead, Block, CausalSelfAttention, FeedForward, LayerNorm

@pytest.fixture()
def tiny_cfg():
    return GPTConfig().validate()

@pytest.fixture()
def tiny_train_cfg():
    return TrainConfig().validate()

@pytest.fixture()
def tiny_cfg_modules():
    return GPTConfig().validate()

@pytest.fixture()
def model(tiny_cfg, tiny_train_cfg):
    torch.manual_seed(0)
    m = HRGPT(tiny_cfg, tiny_train_cfg)
    return m

def test_causal_attention_shapes(model):
    attn = model.transformer.h[0].attn
    x = torch.randn(2, 5, model.config.n_embd)
    y = attn(x)
    assert y.shape == x.shape
    if not attn.flash:
        assert hasattr(attn, "bias") and attn.bias is not None

def test_feedforward_shapes(model):
    ff = model.transformer.h[0].mlp
    x = torch.randn(3, 7, model.config.n_embd)
    y = ff(x)
    assert y.shape == x.shape

def test_forward_and_backward(model, tiny_cfg):
    B, T = 4, 6
    x_batch = torch.randint(0, tiny_cfg.vocab_size, (B, T))
    x_mask = torch.ones_like(x_batch)

    task_names = ["Attrition", "MonthlyIncome", "Attrition", "MonthlyIncome"]
    y_list = [
        torch.tensor(1, dtype=torch.long),
        torch.tensor(5000.0, dtype=torch.float32),
        torch.tensor(0, dtype=torch.long),
        torch.tensor(12000.0, dtype=torch.float32),
    ]

    avg_loss, task_losses = model(x_batch, x_mask, y_list, task_names)
    assert isinstance(avg_loss, torch.Tensor)
    assert "Attrition" in task_losses and "MonthlyIncome" in task_losses
    assert avg_loss.requires_grad

    model.zero_grad(set_to_none=True)
    avg_loss.backward()
    assert model.transformer.wte.weight.grad is not None
    assert any(p.grad is not None for p in model.tasks["Attrition"].parameters())
    assert any(p.grad is not None for p in model.tasks["MonthlyIncome"].parameters())

def test_predict_classification_deterministic(model, tiny_cfg):
    head = model.tasks["Attrition"]
    with torch.no_grad():
        head.fc1.weight.zero_(); head.fc1.bias.zero_()
        head.fc2.weight.zero_()
        head.fc2.bias[:] = torch.tensor([0.0, 10.0])  

    x = torch.randint(0, tiny_cfg.vocab_size, (3, 5))
    out = model.predict(x, "Attrition")
    assert set(out.keys()) == {"pred", "probs", "logits"}
    assert out["pred"].shape == (3,)
    assert torch.all(out["pred"] == 1), "Expected deterministic class 1"

def test_predict_regression_scaling(model, tiny_cfg, tiny_train_cfg):
    head = model.tasks["MonthlyIncome"]
    with torch.no_grad():
        head.fc1.weight.zero_(); head.fc1.bias.zero_()
        head.fc2.weight.zero_(); head.fc2.bias[:] = 2.0
        head.scale[:] = 1.0  

    x = torch.randint(0, tiny_cfg.vocab_size, (2, 5))
    out = model.predict(x, "MonthlyIncome")
    assert set(out.keys()) == {"pred", "value", "raw"}
    assert torch.allclose(out["pred"], torch.full((2,), 2.0 * tiny_train_cfg.reg_unit_value), atol=1e-6)

def test_block_size_assertion(model, tiny_cfg):
    x = torch.randint(0, tiny_cfg.vocab_size, (2, tiny_cfg.block_size + 1))
    x_mask = torch.ones_like(x)
    y_list = [torch.tensor(0, dtype=torch.long), torch.tensor(0.0, dtype=torch.float32)]
    tnames = ["Attrition", "MonthlyIncome"]
    with pytest.raises(AssertionError):
        model(x, x_mask, y_list, tnames)

def test_dropout_modules_present(model):
    assert isinstance(model.transformer.drop, nn.Dropout)
    attn = model.transformer.h[0].attn
    assert isinstance(attn.attn_dropout, nn.Dropout)
    assert isinstance(attn.resid_dropout, nn.Dropout)
    ff = model.transformer.h[0].mlp
    assert isinstance(ff.dropout, nn.Dropout)
    assert isinstance(model.tasks["Attrition"].drop, nn.Dropout)
    assert isinstance(model.tasks["MonthlyIncome"].drop, nn.Dropout)

def test_one_optim_step(model, tiny_cfg, tiny_train_cfg):
    opt = model.get_optimizer()
    B, T = 3, 5
    x = torch.randint(0, tiny_cfg.vocab_size, (B, T))
    m = torch.ones_like(x)
    y_list = [torch.tensor(1, dtype=torch.long),
              torch.tensor(3000.0, dtype=torch.float32),
              torch.tensor(0, dtype=torch.long)]
    tnames = ["Attrition", "MonthlyIncome", "Attrition"]
    loss, _ = model(x, m, y_list, tnames)
    opt.zero_grad(set_to_none=True)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), tiny_train_cfg.grad_norm_clip)
    opt.step()
    assert torch.isfinite(model.transformer.wte.weight).all()

def test_layernorm_equivalence_and_gradients():
    ln = LayerNorm(ndim=6, bias=True, eps=1e-5)
    x = torch.randn(4, 6, requires_grad=True)

    with torch.no_grad():
        ln.weight.fill_(1.0)
        ln.bias.fill_(0.0)

    y_custom = ln(x)
    y_ref = F.layer_norm(x, (6,), ln.weight, ln.bias, eps=ln.eps)
    assert torch.allclose(y_custom, y_ref, atol=1e-6, rtol=1e-5)

    m = y_custom.mean(dim=-1)
    v = y_custom.var(dim=-1, unbiased=False)
    assert torch.allclose(m, torch.zeros_like(m), atol=1e-4)
    assert torch.allclose(v, torch.ones_like(v), atol=1e-3)

    loss = y_custom.sum()
    loss.backward()
    assert x.grad is not None and torch.isfinite(x.grad).all()

def test_feedforward_shape_and_train_vs_eval(tiny_cfg_modules):
    ff = FeedForward(tiny_cfg_modules)
    x = torch.randn(3, 5, tiny_cfg_modules.n_embd)

    y = ff(x)
    assert y.shape == x.shape

    ff.eval()
    with torch.no_grad():
        y1 = ff(x)
        y2 = ff(x)
    assert torch.allclose(y1, y2, atol=1e-7)

    ff.train()
    y3 = ff(x)
    y4 = ff(x)
    if tiny_cfg_modules.resid_pdrop > 0:
        assert not torch.allclose(y3, y4, atol=1e-6)

    loss = y.mean()
    loss.backward()
    assert any(p.grad is not None for p in ff.parameters())

def test_causal_attention_shapes_and_mask(tiny_cfg_modules):
    attn = CausalSelfAttention(tiny_cfg_modules)
    x = torch.randn(2, 6, tiny_cfg_modules.n_embd)
    y = attn(x)
    assert y.shape == x.shape
    if not attn.flash:
        assert hasattr(attn, "bias") and attn.bias is not None

def test_causal_attention_is_actually_causal(tiny_cfg_modules):
    attn = CausalSelfAttention(tiny_cfg_modules)
    attn.eval()  

    B, T, D = 2, 8, tiny_cfg_modules.n_embd
    x1 = torch.randn(B, T, D)
    x2 = x1.clone()

    i = 3
    x2[:, i+1:, :] += torch.randn(B, T - (i + 1), D)

    with torch.no_grad():
        y1 = attn(x1)
        y2 = attn(x2)

    assert torch.allclose(y1[:, : i + 1, :], y2[:, : i + 1, :], atol=1e-5, rtol=1e-4)

def test_block_residual_and_manual_equivalence(tiny_cfg_modules):
    block = Block(tiny_cfg_modules)
    block.eval()  

    x0 = torch.randn(2, 7, tiny_cfg_modules.n_embd, requires_grad=True)

    with torch.no_grad():
        y_block = block(x0)
        
        r1 = block.attn(block.ln_1(x0))
        x1 = x0 + r1
        r2 = block.mlp(block.ln_2(x1))
        y_manual = x1 + r2

    assert torch.allclose(y_block, y_manual, atol=1e-6, rtol=1e-5)

def test_block_gradient_flow(tiny_cfg_modules):
    block = Block(tiny_cfg_modules)
    x = torch.randn(3, 5, tiny_cfg_modules.n_embd, requires_grad=True)
    y = block(x).sum()
    y.backward()
    assert x.grad is not None and torch.isfinite(x.grad).all()
    assert any(p.grad is not None for p in block.parameters())

def test_classification_head_shapes():
    head = ClassificationHead(d_model=16, n_classes=3, mult_fact=2, dropout=0.1, bias=True)
    x = torch.randn(5, 16)
    logits = head(x)
    assert logits.shape == (5, 3)

def test_regression_head_shapes():
    head = RegressionHead(d_model=16, mult_fact=2, dropout=0.1, bias=True)
    x = torch.randn(7, 16)
    output = head(x)
    assert output.shape == (7, 1)

def test_regression_head_scale_clamping():
    torch.manual_seed(0)
    head = RegressionHead(d_model=8, mult_fact=2, dropout=0.0, bias=True).eval()

    x = torch.randn(3, 8)

    with torch.no_grad():
        head.scale.fill_(1.0)
        y_ref = head(x)  
        assert y_ref.shape == (3, 1)

        head.scale.fill_(5.0)
        y_hi = head(x)
        assert torch.allclose(y_hi, y_ref * 2.0, rtol=1e-5, atol=1e-6)

        head.scale.fill_(0.1)
        y_lo = head(x)
        assert torch.allclose(y_lo, y_ref * 0.5, rtol=1e-5, atol=1e-6)

def test_gelu_activation():
    from model.model import GELU
    gelu = GELU()
    x = torch.tensor([-2.0, -1.0, 0.0, 1.0, 2.0])
    y = gelu(x)
    expected = 0.5 * x * (1.0 + torch.tanh(torch.sqrt(torch.tensor(2.0 / torch.pi)) * (x + 0.044715 * torch.pow(x, 3))))
    assert torch.allclose(y, expected, atol=1e-6)

def test_model_forward_input_dimensions(model, tiny_cfg):
    B, T = 2, 8
    x_batch = torch.randint(0, tiny_cfg.vocab_size, (B, T))
    x_mask = torch.ones_like(x_batch)
    
    assert x_batch.shape == (B, T)
    assert x_mask.shape == (B, T)
    
    y_list = [torch.tensor(1, dtype=torch.long), torch.tensor(5000.0, dtype=torch.float32)]
    task_names = ["Attrition", "MonthlyIncome"]
    
    avg_loss, task_losses = model(x_batch, x_mask, y_list, task_names)
    assert isinstance(avg_loss, torch.Tensor)
    assert avg_loss.ndim == 0

def test_transformer_embedding_dimensions(model, tiny_cfg):
    B, T = 3, 10
    x = torch.randint(0, tiny_cfg.vocab_size, (B, T))
    
    tok_emb = model.transformer.wte(x)
    assert tok_emb.shape == (B, T, tiny_cfg.n_embd)
    
    pos = torch.arange(0, T, dtype=torch.long).unsqueeze(0)
    pos_emb = model.transformer.wpe(pos)
    assert pos_emb.shape == (1, T, tiny_cfg.n_embd)

def test_block_dimensions(tiny_cfg_modules):
    block = Block(tiny_cfg_modules)
    B, T = 4, 12
    x = torch.randn(B, T, tiny_cfg_modules.n_embd)
    
    y = block(x)
    assert y.shape == (B, T, tiny_cfg_modules.n_embd)
    
    attn_out = block.attn(block.ln_1(x))
    assert attn_out.shape == (B, T, tiny_cfg_modules.n_embd)
    
    mlp_out = block.mlp(block.ln_2(x))
    assert mlp_out.shape == (B, T, tiny_cfg_modules.n_embd)