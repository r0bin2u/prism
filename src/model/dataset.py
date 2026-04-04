from __future__ import annotations

import json
from pathlib import Path

import torch
from torch.utils.data import Dataset


class ABSADataset(Dataset):
    def __init__(self, data_paths: list[Path], tokenizer, max_length=256):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.samples = []

        for path in data_paths:
            if not path.exists():
                continue
            with open(path) as f:
                for line in f:
                    if not line.strip():
                        continue
                    record = json.loads(line)
                    text = record["text"]
                    source = record["source"]
                    for sl in record["soft_labels"]:
                        self.samples.append({
                            "text": text,
                            "aspect": sl["aspect"],
                            "label": sl["label"],  # [pos, neg, neu]
                            "is_soft_label": source == "llm",
                            "sample_weight": sl["sample_weight"],
                        })

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        s = self.samples[idx]

        # input: "{review_text} [SEP] {aspect_name}"
        encoding = self.tokenizer(
            s["text"],
            s["aspect"],
            truncation=True,
            max_length=self.max_length,
            padding=False,
            return_tensors=None,
        )

        return {
            "input_ids": encoding["input_ids"],
            "attention_mask": encoding["attention_mask"],
            "label": torch.tensor(s["label"], dtype=torch.float32),
            "is_soft_label": s["is_soft_label"],
            "sample_weight": s["sample_weight"],
        }


def collate_fn(batch):
    # pad input_ids and attention_mask to max length in batch
    max_len = max(len(b["input_ids"]) for b in batch)

    input_ids = []
    attention_mask = []
    for b in batch:
        pad_len = max_len - len(b["input_ids"])
        input_ids.append(b["input_ids"] + [0] * pad_len)
        attention_mask.append(b["attention_mask"] + [0] * pad_len)

    return {
        "input_ids": torch.tensor(input_ids, dtype=torch.long),
        "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
        "label": torch.stack([b["label"] for b in batch]),
        "is_soft_label": torch.tensor([b["is_soft_label"] for b in batch], dtype=torch.bool),
        "sample_weight": torch.tensor([b["sample_weight"] for b in batch], dtype=torch.float32),
    }
