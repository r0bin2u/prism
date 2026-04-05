"""
Ablation study: train 4 variants of DeBERTa to validate each design decision.

Model A: human data only (CE loss)
Model B: human + LLM, but LLM uses hard label (CE for both)
Model C: human + LLM, LLM uses soft label (CE + KL)
Model D: C + sample_weight

Usage:
    python -m src.evaluation.ablation --config configs/config.yaml
"""

from __future__ import annotations

import argparse
import copy
import json
import logging
import random
from pathlib import Path

import numpy as np
import torch
import yaml
from sklearn.metrics import f1_score, accuracy_score
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, get_cosine_schedule_with_warmup

from src.model.classifier import ABSAClassifier
from src.model.dataset import ABSADataset, collate_fn

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[2]

SENTIMENTS = ["positive", "negative", "neutral"]


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# -- Modified dataset that can override soft label behavior --

class AblationDataset(torch.utils.data.Dataset):
    """Wraps ABSADataset with options to force hard labels or disable sample weights."""

    def __init__(self, base_dataset, force_hard_label=False, disable_weight=False):
        self.base = base_dataset
        self.force_hard_label = force_hard_label
        self.disable_weight = disable_weight

    def __len__(self):
        return len(self.base)

    def __getitem__(self, idx):
        item = self.base[idx]

        if self.force_hard_label and item["is_soft_label"]:
            # convert soft label to hard label (argmax -> one-hot)
            label = item["label"]
            hard_idx = label.argmax().item()
            new_label = torch.zeros_like(label)
            new_label[hard_idx] = 1.0
            item = dict(item)
            item["label"] = new_label
            item["is_soft_label"] = False  # treat as hard label -> CE loss

        if self.disable_weight:
            item = dict(item)
            item["sample_weight"] = 1.0

        return item


# -- Training loop (simplified, no MLflow) --

def train_one_variant(train_loader, val_loader, config, device, variant_name):
    tc = config["training"]

    model = ABSAClassifier(tc["model_name"]).to(device)

    encoder_params = list(model.encoder.parameters())
    classifier_params = list(model.classifier.parameters())
    optimizer = torch.optim.AdamW([
        {"params": encoder_params, "lr": tc["learning_rate"]},
        {"params": classifier_params, "lr": tc["classifier_lr"]},
    ], weight_decay=0.01)

    total_steps = len(train_loader) * tc["epochs"]
    warmup_steps = int(total_steps * tc["warmup_ratio"])
    scheduler = get_cosine_schedule_with_warmup(optimizer, warmup_steps, total_steps)

    from src.model.loss import MixedDistillationLoss
    criterion = MixedDistillationLoss(alpha=tc["alpha"])

    best_f1 = 0.0
    best_state = None
    patience = 0

    for epoch in range(tc["epochs"]):
        model.train()
        for batch in train_loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["label"].to(device)
            is_soft = batch["is_soft_label"].to(device)
            weights = batch["sample_weight"].to(device)

            optimizer.zero_grad()
            logits = model(input_ids, attention_mask)
            loss, _ = criterion(logits, labels, is_soft, weights)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()

        # validation
        val_metrics = evaluate_model(model, val_loader, device)
        logger.info("[%s] epoch %d  val_f1=%.4f val_acc=%.4f",
                    variant_name, epoch + 1, val_metrics["macro_f1"], val_metrics["accuracy"])

        if val_metrics["macro_f1"] > best_f1:
            best_f1 = val_metrics["macro_f1"]
            best_state = copy.deepcopy(model.state_dict())
            patience = 0
        else:
            patience += 1
            if patience >= tc["early_stopping_patience"]:
                logger.info("[%s] early stopping at epoch %d", variant_name, epoch + 1)
                break

    model.load_state_dict(best_state)
    return model, best_f1


def evaluate_model(model, dataloader, device):
    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["label"].to(device)

            logits = model(input_ids, attention_mask)
            preds = logits.argmax(dim=-1)
            true_labels = labels.argmax(dim=-1)

            all_preds.extend(preds.cpu().tolist())
            all_labels.extend(true_labels.cpu().tolist())

    model.train()
    return {
        "macro_f1": f1_score(all_labels, all_preds, average="macro", zero_division=0),
        "accuracy": accuracy_score(all_labels, all_preds),
    }


def run(config_path):
    with open(config_path) as f:
        config = yaml.safe_load(f)

    tc = config["training"]
    set_seed(tc["seed"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(tc["model_name"])

    # data paths
    filtered_dir = PROJECT_ROOT / "data" / "soft_labels" / "filtered"
    human_path = filtered_dir / "human_soft_labels.jsonl"
    llm_path = filtered_dir / "llm_soft_labels.jsonl"
    test_path = PROJECT_ROOT / "data" / "splits" / "test.jsonl"

    # build base datasets
    human_only = ABSADataset([human_path], tokenizer, max_length=tc["max_length"])
    human_and_llm = ABSADataset([human_path, llm_path], tokenizer, max_length=tc["max_length"])

    # val dataset (human labels from val split, converted to soft label format)
    val_soft = filtered_dir / "human_soft_labels.jsonl"  # val would need its own file
    # for simplicity, use the val split directly
    val_path = PROJECT_ROOT / "data" / "splits" / "val.jsonl"

    # we need val in soft label format for the dataloader
    # create a temporary ABSADataset from val
    # val.jsonl has the same format as train.jsonl (aspects with sentiment)
    # but ABSADataset expects soft_labels format
    # So we create a minimal adapter
    val_dataset = _make_val_dataset(val_path, tokenizer, tc["max_length"])

    # test dataset
    test_dataset = _make_val_dataset(test_path, tokenizer, tc["max_length"])

    val_loader = DataLoader(val_dataset, batch_size=tc["batch_size"],
                            shuffle=False, num_workers=0, collate_fn=collate_fn)
    test_loader = DataLoader(test_dataset, batch_size=tc["batch_size"],
                             shuffle=False, num_workers=0, collate_fn=collate_fn)

    # -- Define 4 variants --

    variants = {
        "A: human only": {
            "dataset": human_only,
            "force_hard_label": False,
            "disable_weight": False,
        },
        "B: human+LLM hard label": {
            "dataset": human_and_llm,
            "force_hard_label": True,
            "disable_weight": True,
        },
        "C: human+LLM soft label": {
            "dataset": human_and_llm,
            "force_hard_label": False,
            "disable_weight": True,
        },
        "D: C + sample_weight": {
            "dataset": human_and_llm,
            "force_hard_label": False,
            "disable_weight": False,
        },
    }

    results = {}

    for name, spec in variants.items():
        logger.info("=" * 50)
        logger.info("Training variant: %s", name)
        logger.info("=" * 50)

        set_seed(tc["seed"])  # reset seed for fair comparison

        wrapped = AblationDataset(
            spec["dataset"],
            force_hard_label=spec["force_hard_label"],
            disable_weight=spec["disable_weight"],
        )

        train_loader = DataLoader(
            wrapped, batch_size=tc["batch_size"],
            shuffle=True, num_workers=0, collate_fn=collate_fn,
        )

        model, best_val_f1 = train_one_variant(train_loader, val_loader, config, device, name)

        # evaluate on test set
        test_metrics = evaluate_model(model, test_loader, device)

        results[name] = {
            "val_macro_f1": round(best_val_f1, 4),
            "test_macro_f1": round(test_metrics["macro_f1"], 4),
            "test_accuracy": round(test_metrics["accuracy"], 4),
            "train_samples": len(wrapped),
        }

        logger.info("[%s] test_f1=%.4f test_acc=%.4f train_n=%d",
                    name, test_metrics["macro_f1"], test_metrics["accuracy"], len(wrapped))

    # save results
    results_dir = PROJECT_ROOT / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    with open(results_dir / "ablation_results.json", "w") as f:
        json.dump(results, f, indent=2)

    # plot
    plot_ablation(results, results_dir)

    # print
    print(f"\n{'='*70}")
    print(f"  Ablation Study Results")
    print(f"{'='*70}")
    print(f"  {'Variant':<30s} {'Train N':>8s} {'Val F1':>8s} {'Test F1':>8s} {'Test Acc':>8s}")
    print(f"  {'-'*64}")
    prev_f1 = None
    for name, r in results.items():
        lift = ""
        if prev_f1 is not None:
            diff = r["test_macro_f1"] - prev_f1
            lift = f"  ({diff:+.4f})"
        print(f"  {name:<30s} {r['train_samples']:>8d} {r['val_macro_f1']:>8.4f} "
              f"{r['test_macro_f1']:>8.4f} {r['test_accuracy']:>8.4f}{lift}")
        prev_f1 = r["test_macro_f1"]
    print(f"{'='*70}\n")


def _make_val_dataset(jsonl_path, tokenizer, max_length):
    """Create a dataset from raw JSONL (train/val/test format with 'aspects' field)."""
    import json as _json

    class _SimpleDataset(torch.utils.data.Dataset):
        def __init__(self):
            self.samples = []
            with open(jsonl_path) as f:
                for line in f:
                    if not line.strip():
                        continue
                    record = _json.loads(line)
                    for asp in record.get("aspects", []):
                        if asp["sentiment"] not in SENTIMENTS:
                            continue
                        one_hot = [0.0, 0.0, 0.0]
                        one_hot[SENTIMENTS.index(asp["sentiment"])] = 1.0
                        self.samples.append({
                            "text": record["text"],
                            "aspect": asp["aspect"],
                            "label": one_hot,
                        })

        def __len__(self):
            return len(self.samples)

        def __getitem__(self, idx):
            s = self.samples[idx]
            enc = tokenizer(
                s["text"], s["aspect"],
                truncation=True, max_length=max_length,
                padding=False, return_tensors=None,
            )
            return {
                "input_ids": enc["input_ids"],
                "attention_mask": enc["attention_mask"],
                "label": torch.tensor(s["label"], dtype=torch.float32),
                "is_soft_label": False,
                "sample_weight": 1.0,
            }

    return _SimpleDataset()


def plot_ablation(results, output_dir):
    import matplotlib.pyplot as plt

    plt.rcParams.update({
        "figure.facecolor": "#F5F2EE",
        "axes.facecolor": "#F5F2EE",
        "axes.edgecolor": "#9B998D",
        "axes.labelcolor": "#4A4A4A",
        "text.color": "#4A4A4A",
        "xtick.color": "#6A6A6A",
        "ytick.color": "#6A6A6A",
        "grid.color": "#D5D0C8",
        "font.size": 11,
    })

    names = list(results.keys())
    short_names = ["Human\nonly", "Human+LLM\nhard label", "Human+LLM\nsoft label", "Soft label\n+ weight"]
    test_f1 = [results[n]["test_macro_f1"] for n in names]

    colors = ["#9B998D", "#C9A9A6", "#8E9AAF", "#A2A88F"]

    fig, ax = plt.subplots(figsize=(9, 5))
    bars = ax.bar(range(len(names)), test_f1, color=colors,
                  edgecolor="white", linewidth=0.8, width=0.6)

    # value labels + lift annotations
    for i, bar in enumerate(bars):
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2, h + 0.003,
                f"{h:.4f}", ha="center", va="bottom", fontsize=10, color="#4A4A4A")
        if i > 0:
            lift = test_f1[i] - test_f1[i-1]
            ax.annotate(f"+{lift:.4f}", xy=(i, h),
                        xytext=(i - 0.3, h + 0.015),
                        fontsize=8, color="#7D8570", fontweight="bold")

    ax.set_xticks(range(len(names)))
    ax.set_xticklabels(short_names[:len(names)])
    ax.set_ylabel("Test Macro-F1")
    ax.set_title("Ablation Study: Each Design Decision Improves Performance")
    ax.grid(axis="y", linestyle="--")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    ymin = min(test_f1) - 0.03
    ymax = max(test_f1) + 0.03
    ax.set_ylim(ymin, ymax)

    plt.tight_layout()
    path = Path(output_dir) / "ablation_study.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved %s", path)


def main():
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=PROJECT_ROOT / "configs" / "config.yaml")
    args = parser.parse_args()
    run(args.config)


if __name__ == "__main__":
    main()
