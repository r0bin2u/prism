from __future__ import annotations

import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer


class ABSAClassifier(nn.Module):
    def __init__(self, model_name="microsoft/deberta-v3-base", num_labels=3):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(model_name)
        hidden_size = self.encoder.config.hidden_size
        self.classifier = nn.Linear(hidden_size, num_labels)

    def forward(self, input_ids, attention_mask):
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        # use [CLS] token representation
        cls_output = outputs.last_hidden_state[:, 0, :]
        logits = self.classifier(cls_output)
        return logits

    @torch.no_grad()
    def predict(self, text, aspect, tokenizer, device="cpu"):
        self.eval()
        encoding = tokenizer(
            text, aspect,
            truncation=True, max_length=256,
            return_tensors="pt",
        )
        input_ids = encoding["input_ids"].to(device)
        attention_mask = encoding["attention_mask"].to(device)

        logits = self.forward(input_ids, attention_mask)
        probs = torch.softmax(logits, dim=-1).squeeze(0)

        sentiments = ["positive", "negative", "neutral"]
        idx = probs.argmax().item()
        return sentiments[idx], probs[idx].item()
