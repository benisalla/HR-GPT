import os
from dataset.tokenizer import MyTokenizer
from dataset.BatchLoader import BatchLoader
from dataclasses import dataclass
from model.config import GPTConfig, DatasetConfig
import torch


# Config
device = "cuda" if torch.cuda.is_available() else "cpu"

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

# Create dataloader
loader = BatchLoader(config, "train", tokenizer, device="cpu")

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


    if batch_idx >= 2:
        print("Stopping after 3 batches for brevity.")
        break
