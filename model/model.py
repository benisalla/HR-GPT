import json
import math
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from dataset.tokenizer import MyTokenizer
from dataset.utils import ORDERED_ATTRS, _pretty_attr
import torch
import torch.nn as nn
from model.config import GPTConfig, TrainConfig
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
    def __init__(self, config: GPTConfig, tr_config: TrainConfig):
        super().__init__()
        assert config.vocab_size is not None
        assert config.block_size is not None
        self.block_size = config.block_size
        self.config = config
        self.tr_config = tr_config

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

    def get_pretrained_backbone(self):
        self_sd = self.state_dict()
        print("Loading pretrained GPT-2 backbone...")
        hf = GPT2LMHeadModel.from_pretrained("gpt2")
        hf_sd = hf.state_dict()

        # token embeddings ( a bit trichy here :) )
        wte_hf = hf_sd["transformer.wte.weight"]                       # [50257, D]
        wte_self = self_sd["transformer.wte.weight"]                   # [new_vocab, D]
        old_vocab, new_vocab = wte_hf.shape[0], wte_self.shape[0]

        with torch.no_grad():
            if new_vocab >= old_vocab:
                wte_self[:old_vocab].copy_(wte_hf)
                if new_vocab > old_vocab:
                    avg = wte_hf.mean(dim=0, keepdim=True)             # [1, D]
                    wte_self[old_vocab:new_vocab].copy_(avg.expand(new_vocab - old_vocab, -1))
            else:
                wte_self.copy_(wte_hf[:new_vocab])

        # the same for block size ( position embeddings or encodings :) )
        if "transformer.wpe.weight" in self_sd and "transformer.wpe.weight" in hf_sd:
            if self_sd["transformer.wpe.weight"].shape == hf_sd["transformer.wpe.weight"].shape:
                self_sd["transformer.wpe.weight"].copy_(hf_sd["transformer.wpe.weight"])

        # copy the rest of the transformer weights
        transposed = (
            "attn.c_attn.weight",
            "attn.c_proj.weight",
            "mlp.c_fc.weight",
            "mlp.c_proj.weight",
        )
        copied, skipped = 0, 0
        for k, v_hf in hf_sd.items():
            # this condition to skip heads ( task heads i added )
            if not k.startswith("transformer."):              
                continue
            if k == "transformer.wte.weight" or k == "transformer.wpe.weight":
                copied += 1
                continue
            if k not in self_sd:
                skipped += 1
                continue
            v_self = self_sd[k]
            try:
                if any(k.endswith(suf) for suf in transposed):
                    # transpose copy
                    if v_hf.shape[::-1] != v_self.shape:
                        raise AssertionError(f"shape {v_hf.shape} vs {v_self.shape}")
                    with torch.no_grad():
                        v_self.copy_(v_hf.t())
                else:
                    if v_hf.shape != v_self.shape:
                        raise AssertionError(f"shape {v_hf.shape} vs {v_self.shape}")
                    with torch.no_grad():
                        v_self.copy_(v_hf)
                copied += 1
            except AssertionError as e:
                skipped += 1
                print(f"[skip] {k}: {e}")  # keep quiet or log if needed

        # load back into the module
        self.load_state_dict(self_sd, strict=False)
        print(f"Backbone copy done. Copied {copied}, skipped {skipped}. Old vocab={old_vocab}, New vocab={new_vocab} (new rows init as avg).")

    def load_weights(
        self,
        ckpt_path: str,
        device: str | torch.device | None = None,
        strict_shapes: bool = True,
        verbose: bool = True,
    ):
        assert os.path.isfile(ckpt_path), f"Checkpoint not found: {ckpt_path}"
        map_loc = device if device is not None else "cpu"
        blob = torch.load(ckpt_path, map_location=map_loc)

        state = blob.get("model", blob)  # support raw state_dict or Trainer checkpoint
        current = self.state_dict()

        if strict_shapes:
            # filter out any key with mismatched shape
            filtered = {k: v for k, v in state.items() if (k in current and current[k].shape == v.shape)}
        else:
            filtered = {k: v for k, v in state.items() if k in current}

        # actually load
        incompatible = self.load_state_dict(filtered, strict=False)
        loaded_keys = list(filtered.keys())

        info = {
            "missing": list(incompatible.missing_keys),
            "unexpected": list(incompatible.unexpected_keys),
            "loaded": loaded_keys,
        }
        if verbose:
            print(f"[load_weights] loaded {len(loaded_keys)} tensors from {ckpt_path}")
            if info["missing"]:
                print(f"[load_weights] missing keys: {len(info['missing'])} (e.g., {info['missing'][:5]})")
            if info["unexpected"]:
                print(f"[load_weights] unexpected keys: {len(info['unexpected'])} (e.g., {info['unexpected'][:5]})")
        return info

    def get_optimizer(self) -> torch.optim.Optimizer:
        decay = set()
        no_decay = set()
        whitelist_weight_modules = (torch.nn.Linear,)
        blacklist_weight_modules = (torch.nn.LayerNorm, torch.nn.Embedding, LayerNorm)

        # classify decay/no_decay
        for mn, m in self.named_modules():
            for pn, p in m.named_parameters():
                fpn = '%s.%s' % (mn, pn) if mn else pn  
                if pn.endswith('bias'):
                    no_decay.add(fpn)
                elif pn.endswith('weight') and isinstance(m, whitelist_weight_modules):
                    decay.add(fpn)
                elif pn.endswith('weight') and isinstance(m, blacklist_weight_modules):
                    no_decay.add(fpn)
                elif pn == 'scale':  # Handle the scale parameter from RegressionHead
                    no_decay.add(fpn)

        # Validate Classification
        param_dict = {pn: p for pn, p in self.named_parameters()}
        inter_params = decay & no_decay
        union_params = decay | no_decay
        assert len(inter_params) == 0, f"Parameters {str(inter_params)} are in both decay/no_decay sets!"
        assert len(param_dict.keys() - union_params) == 0, f"Parameters {str(param_dict.keys() - union_params)} were not classified!"

        # Create the optimizer with separate weight decay for different parameter groups
        optim_groups = [
            {"params": [param_dict[pn] for pn in decay], "weight_decay": self.tr_config.weight_decay},
            {"params": [param_dict[pn] for pn in no_decay], "weight_decay": 0.0},
        ]
        optimizer = torch.optim.AdamW(optim_groups, lr=self.tr_config.learning_rate, betas=self.tr_config.betas)
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
                loss = F.cross_entropy(logits, y_task, label_smoothing=self.tr_config.lbl_smoothing) * self.tr_config.cls_loss_scale
            elif spec.task_type == "regression":
                loss = F.mse_loss(
                    logits.squeeze(-1) / self.tr_config.reg_unit_value, 
                    y_task / self.tr_config.reg_unit_value
                    ) * self.tr_config.reg_loss_scale
                # loss = F.smooth_l1_loss(
                #     logits.squeeze(-1) / self.tr_config.reg_unit_value,
                #     y_task / self.tr_config.reg_unit_value,
                #     beta=1.0  
                # ) * self.tr_config.reg_loss_scale

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
            value_scaled = logits.squeeze(-1)                        # [B] in "thousands"
            value = value_scaled * self.tr_config.reg_unit_value          # back to original units
            return {"pred": value, "value": value, "raw": logits}
        
        else:
            raise ValueError(f"Unknown task_type: {spec.task_type}")
        

# this class will help us in inference
class HRGPTInterface:
    def __init__(self, model_path: Optional[str] = None, data_dir: str = "toy_data"):
        """Initialize the HR-GPT interface"""
        self.data_dir = Path(data_dir)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # Load configuration and model
        self.config = GPTConfig().validate()
        self.tokenizer = MyTokenizer()
        self.config.vocab_size = self.tokenizer.vocab_size
        
        # Initialize model
        self.model = HRGPT(self.config).to(self.device)
        
        # Load pretrained weights if provided
        if model_path and os.path.exists(model_path):
            self._load_model_weights(model_path)
        
        # Load test data for selection
        self.test_data = self._load_test_data()
        
        # Define class mappings for display
        self.class_mappings = {
            "Attrition": {0: "No (Staying)", 1: "Yes (Leaving)"},
            "JobLevel": {0: "Level 1", 1: "Level 2", 2: "Level 3", 3: "Level 4", 4: "Level 5"},
            "JobSatisfaction": {0: "Low", 1: "Medium", 2: "High", 3: "Very High"}
        }
        
        # Define input field configurations
        self.input_fields = self._get_input_fields()
    
    def _load_model_weights(self, model_path: str):
        """Load pretrained model weights"""
        try:
            checkpoint = torch.load(model_path, map_location=self.device)
            if "model" in checkpoint:
                self.model.load_state_dict(checkpoint["model"])
            else:
                self.model.load_state_dict(checkpoint)
            print(f"✅ Loaded model weights from {model_path}")
        except Exception as e:
            print(f"⚠️ Could not load model weights: {e}")
    
    def _load_test_data(self) -> List[Dict]:
        """Load test data for selection"""
        test_file = self.data_dir / "test.json"
        if test_file.exists():
            with open(test_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        return []
    
    def _get_input_fields(self) -> Dict[str, Dict]:
        """Define input field configurations"""
        return {
            "Age": {"type": "number", "label": "Age", "value": 30, "minimum": 18, "maximum": 70},
            "BusinessTravel": {"type": "dropdown", "label": "Business Travel", 
                             "choices": ["non-travel", "travel_rarely", "travel_frequently"], "value": "travel_rarely"},
            "DailyRate": {"type": "number", "label": "Daily Rate ($)", "value": 800, "minimum": 100, "maximum": 2000},
            "Department": {"type": "dropdown", "label": "Department",
                         "choices": ["sales", "research & development", "human resources"], "value": "research & development"},
            "DistanceFromHome": {"type": "number", "label": "Distance from Home (km)", "value": 10, "minimum": 1, "maximum": 50},
            "Education": {"type": "dropdown", "label": "Education Level",
                        "choices": ["Below College", "College", "Bachelor", "Master", "Doctor"], "value": "Bachelor"},
            "EducationField": {"type": "dropdown", "label": "Education Field",
                             "choices": ["life sciences", "medical", "marketing", "technical degree", "other", "human resources"], 
                             "value": "life sciences"},
            "EnvironmentSatisfaction": {"type": "dropdown", "label": "Environment Satisfaction",
                                      "choices": ["Low", "Medium", "High", "Very High"], "value": "High"},
            "Gender": {"type": "dropdown", "label": "Gender", "choices": ["male", "female"], "value": "male"},
            "HourlyRate": {"type": "number", "label": "Hourly Rate ($)", "value": 50, "minimum": 20, "maximum": 100},
            "JobInvolvement": {"type": "dropdown", "label": "Job Involvement",
                             "choices": ["Low", "Medium", "High", "Very High"], "value": "High"},
            "JobRole": {"type": "dropdown", "label": "Job Role",
                      "choices": ["sales executive", "research scientist", "laboratory technician", 
                               "manufacturing director", "healthcare representative", "manager", 
                               "sales representative", "research director", "human resources"], 
                      "value": "research scientist"},
            "MaritalStatus": {"type": "dropdown", "label": "Marital Status",
                            "choices": ["single", "married", "divorced"], "value": "married"},
            "MonthlyRate": {"type": "number", "label": "Monthly Rate ($)", "value": 15000, "minimum": 2000, "maximum": 30000},
            "NumCompaniesWorked": {"type": "number", "label": "Number of Companies Worked", "value": 2, "minimum": 0, "maximum": 10},
            "OverTime": {"type": "dropdown", "label": "Works Overtime", "choices": ["no", "yes"], "value": "no"},
            "PercentSalaryHike": {"type": "number", "label": "Percent Salary Hike (%)", "value": 15, "minimum": 10, "maximum": 25},
            "PerformanceRating": {"type": "dropdown", "label": "Performance Rating",
                                "choices": ["Low", "Good", "Excellent", "Outstanding"], "value": "Excellent"},
            "RelationshipSatisfaction": {"type": "dropdown", "label": "Relationship Satisfaction",
                                       "choices": ["Low", "Medium", "High", "Very High"], "value": "High"},
            "StockOptionLevel": {"type": "number", "label": "Stock Option Level", "value": 1, "minimum": 0, "maximum": 3},
            "TotalWorkingYears": {"type": "number", "label": "Total Working Years", "value": 8, "minimum": 0, "maximum": 40},
            "TrainingTimesLastYear": {"type": "number", "label": "Training Times Last Year", "value": 3, "minimum": 0, "maximum": 10},
            "WorkLifeBalance": {"type": "dropdown", "label": "Work-Life Balance",
                              "choices": ["Bad", "Good", "Better", "Best"], "value": "Better"},
            "YearsAtCompany": {"type": "number", "label": "Years at Company", "value": 5, "minimum": 0, "maximum": 40},
            "YearsInCurrentRole": {"type": "number", "label": "Years in Current Role", "value": 3, "minimum": 0, "maximum": 20},
            "YearsSinceLastPromotion": {"type": "number", "label": "Years Since Last Promotion", "value": 1, "minimum": 0, "maximum": 15},
            "YearsWithCurrManager": {"type": "number", "label": "Years with Current Manager", "value": 2, "minimum": 0, "maximum": 15}
        }
    
    def _convert_input_to_model_format(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """Convert UI inputs to model format"""
        converted = inputs.copy()
        
        # Convert categorical values to lowercase and handle special cases
        categorical_mappings = {
            "Education": {"Below College": 0, "College": 1, "Bachelor": 2, "Master": 3, "Doctor": 4},
            "EnvironmentSatisfaction": {"Low": 0, "Medium": 1, "High": 2, "Very High": 3},
            "JobInvolvement": {"Low": 0, "Medium": 1, "High": 2, "Very High": 3},
            "PerformanceRating": {"Low": 0, "Good": 1, "Excellent": 2, "Outstanding": 3},
            "RelationshipSatisfaction": {"Low": 0, "Medium": 1, "High": 2, "Very High": 3},
            "WorkLifeBalance": {"Bad": 0, "Good": 1, "Better": 2, "Best": 3}
        }
        
        for field, value in converted.items():
            if field in categorical_mappings:
                converted[field] = categorical_mappings[field][value]
            elif isinstance(value, str):
                converted[field] = value.lower()
        
        return converted
    
    def _record_to_text(self, inputs: Dict[str, Any], target_col: str) -> str:
        """Convert record to text format for model input"""
        parts = []
        for attr in ORDERED_ATTRS:
            if attr == target_col or attr not in inputs:
                continue
            parts.append(_pretty_attr(attr, inputs[attr]))
        
        body = ", ".join([p for p in parts if p])
        return f"The employee {body}."
    
    def predict_task(self, task_name: str, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """Make prediction for a specific task"""
        try:
            # Convert inputs to model format
            model_inputs = self._convert_input_to_model_format(inputs)
            
            # Get task specification
            task_spec = self.config.tasks[task_name]
            
            # Convert to text and tokenize
            text = self._record_to_text(model_inputs, task_spec.target_col)
            token_ids = self.tokenizer.encode(text)
            
            # Convert to tensor and predict
            x = torch.tensor([token_ids], dtype=torch.long, device=self.device)
            
            with torch.no_grad():
                self.model.eval()
                result = self.model.predict(x, task_name)
            
            # Format output based on task type
            if task_spec.task_type in ("binary", "multiclass"):
                pred_idx = result["pred"].item()
                probs = result["probs"][0].cpu().numpy()
                
                # Get class name for display
                if task_name in self.class_mappings:
                    pred_class = self.class_mappings[task_name][pred_idx]
                    class_probs = {self.class_mappings[task_name][i]: float(prob) 
                                 for i, prob in enumerate(probs)}
                else:
                    pred_class = f"Class {pred_idx}"
                    class_probs = {f"Class {i}": float(prob) for i, prob in enumerate(probs)}
                
                return {
                    "task": task_name,
                    "prediction": pred_class,
                    "confidence": float(probs[pred_idx]),
                    "all_probabilities": class_probs,
                    "raw_prediction": pred_idx
                }
            
            else:  # regression
                value = result["value"].item()
                return {
                    "task": task_name,
                    "prediction": f"${value:,.2f}" if task_name == "MonthlyIncome" else f"{value:.2f}",
                    "raw_value": float(value)
                }
                
        except Exception as e:
            return {"error": f"Prediction failed: {str(e)}"}
    
    def load_test_sample(self, sample_idx: int) -> Tuple[Dict[str, Any], str]:
        """Load a sample from test data"""
        if not self.test_data or sample_idx >= len(self.test_data):
            return {}, "No test data available"
        
        sample = self.test_data[sample_idx]
        info = f"Loaded test sample {sample_idx + 1}/{len(self.test_data)}"
        
        # Convert sample to UI format
        ui_inputs = {}
        for field, config in self.input_fields.items():
            if field in sample:
                value = sample[field]
                if config["type"] == "dropdown" and isinstance(value, (int, float)):
                    # Convert numeric values back to display format
                    if field == "Education":
                        choices = ["Below College", "College", "Bachelor", "Master", "Doctor"]
                        ui_inputs[field] = choices[int(value)] if 0 <= int(value) < len(choices) else choices[0]
                    elif field in ["EnvironmentSatisfaction", "JobInvolvement", "RelationshipSatisfaction"]:
                        choices = ["Low", "Medium", "High", "Very High"]
                        ui_inputs[field] = choices[int(value)] if 0 <= int(value) < len(choices) else choices[0]
                    elif field == "PerformanceRating":
                        choices = ["Low", "Good", "Excellent", "Outstanding"]
                        ui_inputs[field] = choices[int(value)] if 0 <= int(value) < len(choices) else choices[0]
                    elif field == "WorkLifeBalance":
                        choices = ["Bad", "Good", "Better", "Best"]
                        ui_inputs[field] = choices[int(value)] if 0 <= int(value) < len(choices) else choices[0]
                    else:
                        ui_inputs[field] = value
                else:
                    ui_inputs[field] = value
            else:
                ui_inputs[field] = config.get("value", "")
        
        return ui_inputs, info