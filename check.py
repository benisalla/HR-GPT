import os
from dataset.tokenizer import MyTokenizer
from dataset.BatchLoader import BatchLoader
from dataclasses import dataclass
from model.config import GPTConfig, DatasetConfig
from model.model import HRGPT
import torch


print("Check.py is Running ...")
print("torch:", torch.__version__)
print("cuda available:", torch.cuda.is_available())
print("torch cuda:", torch.version.cuda)
if torch.cuda.is_available():
    print("gpu:", torch.cuda.get_device_name(0))


# Config
device = "cuda" if torch.cuda.is_available() else "cpu"

print(f"Using device: {device}")

config = GPTConfig(
    vocab_size=50257,
    block_size=1024,
    n_layer=12,
    n_head=12,
    n_embd=768,
    dataset=DatasetConfig(
        data_dir="toy_data",
        batch_size=4,
        toks_in_batch=1000  # must match what BatchLoader expects
    )
)

# Create tokenizer
tokenizer = MyTokenizer("gpt2")
config.vocab_size = len(tokenizer.tokenizer) 

# Create dataloader
loader = BatchLoader(config, "val", tokenizer, device="cpu")

print(f"Number of batches: {len(loader)}\n")

# Iterate and inspect
for batch_idx, (x_batch, x_mask, y_list, task_names) in enumerate(loader):
    print(f"--- Batch {batch_idx+1}/{len(loader)} ---")
    for i in range(x_batch.size(0)):
        ids = x_batch[i].tolist()
        mask = x_mask[i].tolist()
        # Decode ignoring pads
        decoded = tokenizer.decode([tid for tid, m in zip(ids, mask) if m == 1], skip_special_tokens=False)
        print(f" Sample {i+1}:")
        print(f"   input_ids: {ids}")
        print(f"   mask:      {mask}")
        print(f"   decoded:   {decoded}")
        print(f"   target y:  {y_list[i].item()}")
        print(f"   task:      {task_names[i]}")
    print()


    if batch_idx >= 0:
        print("Stopping after 3 batches for brevity.")
        break


print("Creating model...")

# Create model
model = HRGPT(config).to(device)
model.eval()  # for testing
print("Model initialized.\n")

# Iterate and inspect
for batch_idx, (x_batch, x_mask, y_list, task_names) in enumerate(loader):
    print(f"--- Batch {batch_idx+1}/{len(loader)} ---")

    # Move tensors to device
    x_batch = x_batch.to(device)
    x_mask = x_mask.to(device)
    y_list = [y.to(device) for y in y_list]

    # Forward pass (loss computation)
    with torch.no_grad():
        avg_loss, task_losses = model(x_batch, x_mask, y_list, task_names)

    print(f"   Avg loss: {avg_loss.item():.4f}")
    print(f"   Task losses: {task_losses}")

    # Predict for first sample in batch
    first_task = task_names[0]
    pred_out = model.predict(x_batch[:1], first_task)
    print(f"   Prediction for first sample ({first_task}): {pred_out}")

    if batch_idx >= 2:
        print("Stopping after 3 batches for brevity.")
        break


model.eval()
with torch.no_grad():
    for batch_idx, (x_batch, x_mask, y_list, task_names) in enumerate(loader):
        x_batch = x_batch.to(device)
        x_mask  = x_mask.to(device)

        print(f"\n--- Predict batch {batch_idx+1} ---")
        for i in range(x_batch.size(0)):
            task = task_names[i]
            out  = model.predict(x_batch[i:i+1], task)

            if config.tasks[task].task_type in ("binary", "multiclass"):
                pred = out["pred"].item()
                probs = out["probs"][0].tolist()
                print(f" sample {i}: task={task} -> pred={pred}, probs={probs}")
            else:
                val = out["value"].item()
                print(f" sample {i}: task={task} -> value={val}")

        # keep it short
        if batch_idx >= 2:
            break
