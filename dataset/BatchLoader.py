import json
import torch
from random import shuffle
from itertools import groupby
from torch.nn.utils.rnn import pad_sequence
from dataset.utils import ORDERED_ATTRS, _pretty_attr

class BatchLoader(object):
    def __init__(self, config, split, tokenizer, device="cpu"):
        self.toks_in_batch = int(config.dataset.toks_in_batch)
        self.batch_size = int(config.dataset.batch_size)
        self.split = split
        self.data_dir = config.dataset.data_dir
        self.training = (split == "train")
        self.tokenizer = tokenizer
        self.device = device
        self.tasks = config.tasks 
        self.pad_id = getattr(tokenizer, "pad_id", None) or tokenizer.tokenizer.pad_token_id

        # Load split
        with open(f"{config.dataset.data_dir}/{split}.json", "r", encoding="utf-8") as f:
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

                # Deal with Target Y (Int -> classification or Float -> regression)
                if spec.task_type == "regression":
                    y = torch.tensor(float(inputs[tgt]), dtype=torch.float32)
                else:
                    y = torch.tensor(int(inputs[tgt]), dtype=torch.long)

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
        return f"The employee {body}."

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
        self.idx = -1

    def __len__(self):
        return self.n_batches

    def __iter__(self):
        self.idx = -1
        return self

    def __next__(self):
        self.idx += 1
        if self.idx >= self.n_batches:
            raise StopIteration
        batch = self.batches[self.idx]

        ids_list = [torch.LongTensor(s["ids"]) for s in batch]
        input_ids = pad_sequence(sequences=ids_list, batch_first=True, padding_value=self.pad_id)

        x_batch = input_ids.to(self.device, non_blocking=True)
        x_mask = x_batch.ne(self.pad_id).long()

        y_list = [s["y"].to(self.device, non_blocking=True) for s in batch]
        task_names = [s["task"] for s in batch]
        return x_batch, x_mask, y_list, task_names