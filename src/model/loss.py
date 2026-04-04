from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class MixedDistillationLoss(nn.Module):
    def __init__(self, alpha=0.7):
        super().__init__()
        self.alpha = alpha  # weight for human (CE) loss

    def forward(self, logits, labels, is_soft_label, sample_weights):
        human_mask = ~is_soft_label
        llm_mask = is_soft_label

        # human part: cross entropy
        if human_mask.any():
            human_logits = logits[human_mask]
            human_labels = labels[human_mask].argmax(dim=-1)  # one-hot -> int
            human_weights = sample_weights[human_mask]
            human_loss = F.cross_entropy(human_logits, human_labels, reduction="none")
            human_loss = (human_loss * human_weights).mean()
        else:
            human_loss = torch.tensor(0.0, device=logits.device)

        # llm part: KL divergence
        if llm_mask.any():
            llm_logits = logits[llm_mask]
            llm_labels = labels[llm_mask]
            llm_weights = sample_weights[llm_mask]
            student_log_probs = F.log_softmax(llm_logits, dim=-1)
            kl = F.kl_div(student_log_probs, llm_labels, reduction="none").sum(dim=-1)
            llm_loss = (kl * llm_weights).mean()
        else:
            llm_loss = torch.tensor(0.0, device=logits.device)

        total = self.alpha * human_loss + (1 - self.alpha) * llm_loss

        return total, {
            "human_loss": human_loss.item(),
            "llm_loss": llm_loss.item(),
            "total_loss": total.item(),
        }
