import math
import torch
import torch.nn as nn
from model.config import GPTConfig
from torch.nn import functional as F
from transformers import GPT2LMHeadModel

class GELU(nn.Module):
    "Same as torch.nn.GELU(approximate='tanh')"

    def forward(self, x):
        # constant term
        s2pi = math.sqrt(2.0 / math.pi)  # √(2/π)

        # cubic term for approximation
        xt3  = x + 0.044715 * torch.pow(x, 3)  # x + 0.044715·x³

        # tanh-based approximation
        return 0.5 * x * (1.0 + torch.tanh(s2pi * xt3))
    
class LayerNorm(nn.Module):
    def __init__(self, ndim: int, bias: bool = True, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(ndim))
        if bias:
            self.bias = nn.Parameter(torch.zeros(ndim))
        else:
            self.register_parameter("bias", None)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.layer_norm(x, x.shape[-1:], self.weight, self.bias, self.eps)

class CausalSelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0 ; "n_embd % n_head should be 0."
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd, bias=config.bias)
        self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)

        # regularization
        self.attn_pdrop = config.attn_pdrop
        self.resid_pdrop = config.resid_pdrop
        self.attn_dropout = nn.Dropout(config.attn_pdrop)
        self.resid_dropout = nn.Dropout(config.resid_pdrop)
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.flash = hasattr(torch.nn.functional, 'scaled_dot_product_attention')
        if not self.flash:
            print("WARNING: Using the standard attention mechanism. Flash Attention requires PyTorch version 2.0 or higher.")
            self.register_buffer("bias", torch.tril(torch.ones(config.block_size, config.block_size))
                                        .view(1, 1, config.block_size, config.block_size))

    def forward(self, x):
        B, T, C = x.size()
        q, k, v  = self.c_attn(x).split(self.n_embd, dim=2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)   
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)   
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)   

        if self.flash:
            y = torch.nn.functional.scaled_dot_product_attention(q, k, v, attn_mask=None, dropout_p=self.resid_pdrop if self.training else 0, is_causal=True)
        else:
            att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
            att = att.masked_fill(self.bias[:,:,:T,:T] == 0, float('-inf'))
            att = F.softmax(att, dim=-1)
            att = self.attn_dropout(att)
            y = att @ v  
        y = y.transpose(1, 2).contiguous().view(B, T, C) 

        y = self.resid_dropout(self.c_proj(y))
        return y

class FeedForward(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.c_fc    = nn.Linear(config.n_embd, 4 * config.n_embd, bias=config.bias)
        self.gelu    = GELU()
        self.c_proj  = nn.Linear(4 * config.n_embd, config.n_embd, bias=config.bias)
        self.dropout = nn.Dropout(config.resid_pdrop)

    def forward(self, x):
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        x = self.dropout(x)
        return x

class Block(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.ln_1 = LayerNorm(config.n_embd, bias=config.bias)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = LayerNorm(config.n_embd, bias=config.bias)
        self.mlp = FeedForward(config)

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))    
        x = x + self.mlp(self.ln_2(x))  
        return x

class ClassificationHead(nn.Module):
    """Enhanced MLP head for classification: [B, D] -> [B, n_classes]"""
    def __init__(self, d_model: int, n_classes: int, mult_fact: int, dropout: float, bias: bool):
        super().__init__()
        
        self.fc1 = nn.Linear(d_model, mult_fact * d_model, bias=bias)
        self.ln1 = nn.LayerNorm(mult_fact * d_model)
        self.act = GELU()
        self.drop = nn.Dropout(dropout)
        self.fc2 = nn.Linear(mult_fact * d_model, n_classes, bias=bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.fc1(x)
        h = self.ln1(h)
        h = self.act(h)  
        h = self.drop(h)
        logits = self.fc2(h)
        return logits 

class RegressionHead(nn.Module):
    """Enhanced MLP head for regression: [B, D] -> [B, 1]"""
    def __init__(self, d_model: int, mult_fact: int, dropout: float, bias: bool):
        super().__init__()
        
        self.fc1 = nn.Linear(d_model, mult_fact * d_model, bias=bias)
        self.ln1 = nn.LayerNorm(mult_fact * d_model)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(mult_fact * d_model, 1, bias=bias)
        self.drop = nn.Dropout(dropout)
        self.scale = nn.Parameter(torch.ones(1))  # scaling factor

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.fc1(x)        
        h = self.ln1(h)        
        h = self.act(h)         
        h = self.drop(h)
        y = self.fc2(h)  
        y = y * self.scale
        return y  # [B, 1]

def get_task_heads(cfg: GPTConfig) -> nn.ModuleDict:
    heads = {}
    for name, spec in cfg.tasks.items():
        if spec.task_type in ("binary", "multiclass"):
            heads[name] = ClassificationHead(
                d_model=cfg.n_embd,
                n_classes=spec.n_classes,
                mult_fact=spec.mult_fact,
                dropout=spec.dropout,
                bias=cfg.bias
            )
        elif spec.task_type == "regression":
            heads[name] = RegressionHead(
                d_model=cfg.n_embd,
                mult_fact=spec.mult_fact,
                dropout=spec.dropout,
                bias=cfg.bias
            )
        else:
            raise ValueError(f"Unknown task_type for {name}: {spec.task_type}")
    return nn.ModuleDict(heads)


class HRGPT(nn.Module):
    """Our Humain Resources GPT Model"""
    def __init__(self, config):
        super().__init__()
        assert config.vocab_size is not None
        assert config.block_size is not None
        self.block_size = config.block_size
        self.config = config

        # transformer core
        self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(config.vocab_size, config.n_embd),
            wpe = nn.Embedding(config.block_size, config.n_embd),
            drop = nn.Dropout(config.embd_pdrop),
            h = nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
            ln_f = nn.LayerNorm(config.n_embd),
        ))

        print("Creating task heads...")
        # tasks' heads 
        self.tasks = get_task_heads(config)

        print("applying initial weights ...")
        # init all weights
        self.apply(self.__init_weights__)

        print("Initializing from pretrained GPT-2 weights...")
        # init model
        self.__init_from_pretrained__()

        print("displaying model stats...")
        # display stats of the model and his head
        self.__model_stats__()

    def __model_stats__(self):
        total_params = 0
        
        # transformer params
        transformer_params = sum(p.numel() for p in self.transformer.parameters())
        total_params += transformer_params
        print(f"Transformer parameters: {transformer_params / 1e6:.2f}M")

        # heads params
        for task_name, task_head in self.tasks.items():
            task_params = sum(p.numel() for p in task_head.parameters())
            total_params += task_params
            print(f"Task '{task_name}' head parameters: {task_params / 1e6:.2f}M")
        
        # total params
        print(f"Total parameters: {total_params / 1e6:.2f}M")

    def __init_weights__(self, module):
        for pn, p in self.named_parameters():
            if pn.endswith('c_proj.weight'):
                torch.nn.init.normal_(p, mean=0.0, std=0.02 / math.sqrt(2 * self.config.n_layer))

        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, LayerNorm):
            torch.nn.init.zeros_(module.bias)
            torch.nn.init.ones_(module.weight)

    def __init_from_pretrained__(self):
        # our params
        sd = self.state_dict()

        # load HF GPT-2
        print("Loading pretrained GPT-2 model...")
        hf = GPT2LMHeadModel.from_pretrained("gpt2")
        print(f"Pretrained model loaded with {sum(p.numel() for p in hf.parameters()) / 1e6:.2f}M parameters.")
        sd_hf = hf.state_dict()

        # copy ONLY transformer weights 
        hf_keys = [k for k in sd_hf.keys() if k.startswith("transformer.")]
        transposed = (
            "attn.c_attn.weight",
            "attn.c_proj.weight",
            "mlp.c_fc.weight",
            "mlp.c_proj.weight",
        )

        copied, skipped = 0, 0
        for k in hf_keys:
            if k not in sd:
                skipped += 1
                print(f"[skip] missing in SELF: {k}")
                continue

            v_hf, v_self = sd_hf[k], sd[k]
            try:
                if any(k.endswith(suf) for suf in transposed):
                    assert v_hf.shape[::-1] == v_self.shape, f"Shape mismatch for {k}: {v_hf.shape} vs {v_self.shape}"
                    with torch.no_grad():
                        v_self.copy_(v_hf.t())
                else:
                    assert v_hf.shape == v_self.shape, f"Shape mismatch for {k}: {v_hf.shape} vs {v_self.shape}"
                    with torch.no_grad():
                        v_self.copy_(v_hf)
                copied += 1
            except AssertionError as e:
                skipped += 1
                print(f"[skip] {e}")

        # init our task heads (custom) — keep your original intent
        for _, task_head in self.tasks.items():
            if isinstance(task_head, nn.Module):
                task_head.apply(self.__init_weights__)

        # load back into self (non-transformer keys remain as-initialized)
        self.load_state_dict(sd, strict=False)
        print(f"Copied {copied} transformer tensors from GPT-2, skipped {skipped}.")

    def get_optimizer(self, config):
        decay = set()
        no_decay = set()
        whitelist_weight_modules = (torch.nn.Linear,)
        blacklist_weight_modules = (torch.nn.LayerNorm, torch.nn.Embedding)

        # classify decay/no_decay
        for mn, m in self.named_modules():
            for pn, _ in m.named_parameters():
                fpn = '%s.%s' % (mn, pn) if mn else pn  
                if pn.endswith('bias'):
                    no_decay.add(fpn)
                elif pn.endswith('weight') and isinstance(m, whitelist_weight_modules):
                    decay.add(fpn)
                elif pn.endswith('weight') and isinstance(m, blacklist_weight_modules):
                    no_decay.add(fpn)

        # Validate Classification
        param_dict = {pn: p for pn, p in self.named_parameters()}
        inter_params = decay & no_decay
        union_params = decay | no_decay
        assert len(inter_params) == 0, f"Parameters {str(inter_params)} are in both decay/no_decay sets!"
        assert len(param_dict.keys() - union_params) == 0, f"Parameters {str(param_dict.keys() - union_params)} were not classified!"

        # Create the optimizer with separate weight decay for different parameter groups
        optim_groups = [
            {"params": [param_dict[pn] for pn in decay], "weight_decay": config.weight_decay},
            {"params": [param_dict[pn] for pn in no_decay], "weight_decay": 0.0},
        ]
        optimizer = torch.optim.AdamW(optim_groups, lr=config.learning_rate, betas=config.betas)
        return optimizer

    def forward(self, x_batch, x_mask, y_list, task_names):
        B, T = x_batch.size()
        device = x_batch.device
        assert T <= self.block_size, f"T={T} > block_size={self.block_size}"

        # Shared Transformer
        pos_enc = torch.arange(0, T, dtype=torch.long, device=x_batch.device).unsqueeze(0)
        tok_emb = self.transformer.wte(x_batch)   # (B, T, n_embd)
        pos_emb = self.transformer.wpe(pos_enc)   # (1, T, n_embd)
        x = self.transformer.drop((tok_emb + pos_emb) * x_mask.unsqueeze(-1))

        for block in self.transformer.h:
            x = block(x)
        x = self.transformer.ln_f(x)              # (B, T, n_embd)
        x = x[:, -1, :]                           # (B, n_embd) last token is enough because of masked attention

        # Group by task for batched head forward
        task_to_indices = {}
        for i, tname in enumerate(task_names):
            task_to_indices.setdefault(tname, []).append(i)

        task_avg_loss = {}
        total_loss = 0.0

        for tname, indices in task_to_indices.items():
            idx_tensor = torch.tensor(indices, dtype=torch.long, device=device)
            x_task = x[idx_tensor]                              # (N_task, n_embd)
            y_task = torch.stack([y_list[i] for i in indices])  
            head = self.tasks[tname]
            logits = head(x_task)

            spec = self.config.tasks[tname]
            if spec.task_type in ("binary", "multiclass"):
                loss = F.cross_entropy(logits, y_task)
            elif spec.task_type == "regression":
                loss = F.mse_loss(logits.squeeze(-1), y_task)
            else:
                raise ValueError(f"Unknown task type {spec.task_type}")

            task_avg_loss[tname] = loss.item()
            total_loss += loss

        avg_loss = total_loss / len(task_to_indices)
        return avg_loss, task_avg_loss

    @torch.no_grad()
    def predict(self, x: torch.Tensor, task_name: str):
        self.eval()
        device = x.device
        b, t = x.size()
        assert task_name in self.tasks, f"Unknown task '{task_name}'"
        spec = self.config.tasks[task_name]
        assert t <= self.block_size, f"seq len {t} > block_size {self.block_size}"

        # shared transformer
        pos = torch.arange(0, t, dtype=torch.long, device=device).unsqueeze(0)  # [1, T]
        tok_emb = self.transformer.wte(x)            # [B, T, D]
        pos_emb = self.transformer.wpe(pos)          # [1, T, D]
        h = self.transformer.drop(tok_emb + pos_emb)
        for blk in self.transformer.h:
            h = blk(h)
        h = self.transformer.ln_f(h)                 # [B, T, D]
        h_last = h[:, -1, :]                         # [B, D]

        # task head
        head = self.tasks[task_name]
        logits = head(h_last)                        # [B, C] or [B, 1]

        # post-process per task type
        if spec.task_type in ("binary", "multiclass"):
            probs = F.softmax(logits, dim=-1)        # [B, C]  (binary C=2)
            pred = probs.argmax(dim=-1)              # [B]
            return {"pred": pred, "probs": probs, "logits": logits}

        elif spec.task_type == "regression":
            value = logits.squeeze(-1)               # [B]
            return {"pred": value, "value": value, "raw": logits}

        else:
            raise ValueError(f"Unknown task_type: {spec.task_type}")