from dataclasses import dataclass
from utils import ORDERED_ATTRS, _pretty_attr
import torch
import json
from torch.nn.utils.rnn import pad_sequence
from itertools import groupby
from random import shuffle

class BatchLoader(object):
    """
    Multitask loader that returns:
      x_batch: LongTensor [B, L] (padded token ids of <sost>...<eost>)
      y_list:  list[Tensor] length B (Long for cls, Float for reg)
      task_names: list[str] length B
    """
    def __init__(self, config, data_dir, split, tokenizer, toks_in_batch, device="cpu",
                 min_len=3, max_len=400):
        self.toks_in_batch = int(toks_in_batch)
        self.training = (split == "train")
        self.tokenizer = tokenizer
        self.device = device
        self.min_len = min_len
        self.max_len = max_len
        self.tasks = config.tasks 

        # Load split
        with open(f"{data_dir}/{split}.json", "r", encoding="utf-8") as f:
            raw_data = json.load(f)

        # build samples for each task
        self.samples = []
        for inputs in raw_data:
            for tname, spec in self.tasks.items():
                tgt = spec.target_col
                if tgt not in inputs:
                    raise ValueError(f"Target column '{tgt}' not found in inputs keys.")

                # format the input record to text
                x_text = self.__record_to_text__(inputs, tgt=tgt)
                token_ids = self.tokenizer.encode(x_text)
                n = len(token_ids)
                if n < self.min_len or n > self.max_len:
                    continue

                # Deal with Target Y (Int -> classification or Float -> regression)
                if spec.task_type == "regression":
                    y = torch.tensor([float(inputs[tgt])], dtype=torch.float32)
                else:
                    y = torch.tensor([int(inputs[tgt])], dtype=torch.long)

                self.samples.append({"ids": token_ids, "len": n, "y": y, "task": tname})

        # Sort by length for better packing if training
        if self.training:
            self.samples.sort(key=lambda s: s["len"])

        # Precompute batches
        self.create_batches()

    def __record_to_text__(self, inputs: dict, tgt: str) -> str:
        parts = []
        for attr in ORDERED_ATTRS:
            if attr == tgt or attr not in inputs:
                continue
            parts.append(_pretty_attr(attr, inputs[attr]))
        # remove empties and join nicely
        body = ", ".join([p for p in parts if p])
        # small style tweak: clauses read better after "The employee"
        return f"<sost> The employee {body}. <eost>"

    def create_batches(self):
        if self.training:
            chunks = [list(g) for _, g in groupby(self.samples, key=lambda s: s["len"])]
            self.batches = []
            for chunk in chunks:
                L = chunk[0]["len"]
                seqs_per_batch = max(1, self.toks_in_batch // max(1, L))
                for i in range(0, len(chunk), seqs_per_batch):
                    self.batches.append(chunk[i:i + seqs_per_batch])
            shuffle(self.batches)
        else:
            self.batches = [[s] for s in self.samples]

        self.n_batches = len(self.batches)
        self.current = -1

    def __iter__(self):
        return self

    def __len__(self):
        return len(self.samples)

    def __next__(self):
        self.current += 1
        if self.current >= self.n_batches:
            self.create_batches()
            self.current = 0

        batch = self.batches[self.current]

        ids_list = [torch.LongTensor(s["ids"]) for s in batch]
        input_ids = pad_sequence(
            sequences=ids_list,
            batch_first=True,
            padding_value=self.tokenizer.pad_token_id
        )

        x_batch = input_ids.to(self.device)
        y_list = [s["y"].to(self.device) for s in batch]
        task_names = [s["task"] for s in batch]

        return x_batch, y_list, task_names
