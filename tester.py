import os
from collections import defaultdict
import torch
import matplotlib.pyplot as plt
from model.config import DatasetConfig, GPTConfig, TrainConfig
from model.model import HRGPT
from dataset.tokenizer import MyTokenizer
from dataset.BatchLoader import BatchLoader


@torch.no_grad()
def compute_test_metrics(model, test_loader, device):
    model.eval()

    total_loss = 0.0
    num_batches = 0

    # aggregate per-task losses
    task_losses = defaultdict(list)

    # aggregate per-task metrics
    cls_correct = defaultdict(int)   
    cls_total   = defaultdict(int)
    reg_sse = defaultdict(float)     
    reg_sae = defaultdict(float)     
    reg_cnt = defaultdict(int)       

    for x_batch, x_mask, y_list, task_names in test_loader:
        # to device
        x_batch = x_batch.to(device)
        x_mask  = x_mask.to(device)
        y_list  = [y.to(device) for y in y_list]

        # forward - loss per batch
        avg_loss, batch_task_losses = model(x_batch, x_mask, y_list, task_names)
        total_loss += avg_loss.item()
        for task, loss in batch_task_losses.items():
            task_losses[task].append(loss)
        num_batches += 1

        # group indices by task
        task_to_indices = defaultdict(list)
        for i, tname in enumerate(task_names):
            task_to_indices[tname].append(i)

        # per-task metrics
        for tname, idxs in task_to_indices.items():
            indices = torch.tensor(idxs, dtype=torch.long, device=device)
            x_task  = x_batch.index_select(0, indices)
            y_task  = torch.stack([y_list[i] for i in idxs])

            spec = model.config.tasks[tname]
            out  = model.predict(x_task, tname)

            if spec.task_type in ("binary", "multiclass"):
                # y_task is class indices
                pred = out["pred"]                    # [N]
                cls_correct[tname] += (pred == y_task).sum().item()
                cls_total[tname]   += y_task.numel()
            elif spec.task_type == "regression":
                # y_task and predicted value are floats
                val  = out["value"].view(-1)          # [N]
                yvec = y_task.view(-1)
                diff = val - yvec
                reg_sse[tname] += torch.sum(diff * diff).item()
                reg_sae[tname] += torch.sum(diff.abs()).item()
                reg_cnt[tname]  += yvec.numel()
            else:
                raise ValueError(f"Unknown task type for metrics: {spec.task_type}")

    # finalize
    metrics = {
        "test/total_loss": (total_loss / max(1, num_batches)) if num_batches > 0 else float("nan"),
    }

    # average per-task losses
    for task, losses in task_losses.items():
        if len(losses) > 0:
            metrics[f"test/{task}_loss"] = sum(losses) / len(losses)

    # classification accuracies
    for task, tot in cls_total.items():
        if tot > 0:
            acc = cls_correct[task] / tot
            metrics[f"test/{task}_acc"] = acc * 100.0  # percent

    # regression MSE/MAE
    for task, cnt in reg_cnt.items():
        if cnt > 0:
            mse = reg_sse[task] / cnt
            mae = reg_sae[task] / cnt
            metrics[f"test/{task}_mse"] = mse
            metrics[f"test/{task}_mae"] = mae

    model.train()
    return metrics


def plot_bar(title, items, ylabel):
    """Simple helper to draw a single bar chart from (name->value) dict."""
    if not items:
        return
    names = list(items.keys())
    vals  = [items[n] for n in names]

    plt.figure()
    plt.title(title)
    plt.ylabel(ylabel)
    plt.bar(names, vals)
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()


# run test for our model
# Setup ml_config
dt_config = DatasetConfig(
    data_dir = "toy_data",
    toks_in_batch = 1000,
    batch_size = 16,
    data_stats_path = "data_stats.json",
)
ml_config = GPTConfig()
tr_config = TrainConfig(
    device= "cuda" if torch.cuda.is_available() else "cpu",
)

# Create tokenizer  
tokenizer = MyTokenizer()
ml_config.vocab_size = len(tokenizer.tokenizer)

# Create model
model = HRGPT(ml_config, tr_config)

# Load pretrained weights if available
CKPT_PATH = "./checkpoints/hr_gpt_training/best_model_step_12500.pt.pt"
model.load_trained_model(
    ckpt_path=CKPT_PATH, device=tr_config.device, strict_shapes=True, verbose=True
)
model = model.to(tr_config.device)

# Create datasets
test_loader = BatchLoader(ml_config, "test", tokenizer, device=tr_config.device)

# Compute metrics
metrics = compute_test_metrics(model, test_loader, device=tr_config.device)

# get name from ckeckpoints name
ckpt_title = os.path.basename(CKPT_PATH)
out_file   = f"test_metrics_{ckpt_title}.png".replace(" ", "_")

# Prepare lines
lines = []
for k in sorted(metrics.keys()):
    v = metrics[k]
    try:
        lines.append(f"{k}: {v:.6f}")
    except Exception:
        lines.append(f"{k}: {v}")

# Figure height scales with number of lines
fig_height = max(6, 0.35 * (len(lines) + 6))
plt.figure(figsize=(12, fig_height))
plt.axis("off")
plt.title(ckpt_title, pad=12)

y = 0.96
plt.text(0.02, y, "=== Test Metrics Summary ===", fontsize=12, fontweight="bold", family="monospace")
y -= 0.05
for line in lines:
    plt.text(0.02, y, line, fontsize=10, family="monospace")
    y -= 0.03

plt.tight_layout()
plt.savefig(out_file, dpi=200)
print(f"Saved metrics image to: {out_file}")
