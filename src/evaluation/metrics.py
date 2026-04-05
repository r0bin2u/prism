from __future__ import annotations

import argparse
import json
import logging
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import yaml
from sklearn.metrics import (
    accuracy_score, f1_score, classification_report, confusion_matrix
)
from transformers import AutoTokenizer

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[2]

SENTIMENTS = ["positive", "negative", "neutral"]


def load_test_data(test_path):
    samples = []
    with open(test_path) as f:
        for line in f:
            if not line.strip():
                continue
            record = json.loads(line)
            for asp in record.get("aspects", []):
                if asp["sentiment"] not in SENTIMENTS:
                    continue
                samples.append({
                    "text": record["text"],
                    "aspect": asp["aspect"],
                    "sentiment": asp["sentiment"],
                })
    return samples


def predict_batch(model, tokenizer, samples, device, batch_size=64):
    model.eval()
    all_preds = []
    all_probs = []

    for start in range(0, len(samples), batch_size):
        batch = samples[start:start + batch_size]
        texts = [s["text"] for s in batch]
        aspects = [s["aspect"] for s in batch]

        enc = tokenizer(
            texts, aspects,
            truncation=True, max_length=256,
            padding=True, return_tensors="pt",
        )
        input_ids = enc["input_ids"].to(device)
        attention_mask = enc["attention_mask"].to(device)

        with torch.no_grad():
            logits = model(input_ids, attention_mask)
            probs = torch.softmax(logits, dim=-1).cpu().numpy()

        for i in range(len(batch)):
            pred_idx = probs[i].argmax()
            all_preds.append(SENTIMENTS[pred_idx])
            all_probs.append(probs[i].tolist())

    return all_preds, all_probs


def plot_confusion_matrix(cm, output_path):
    import matplotlib.pyplot as plt
    import seaborn as sns

    plt.rcParams.update({
        "figure.facecolor": "#F5F2EE",
        "axes.facecolor": "#F5F2EE",
        "text.color": "#4A4A4A",
        "font.size": 11,
    })

    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="YlGnBu",
        xticklabels=SENTIMENTS, yticklabels=SENTIMENTS,
        ax=ax, linewidths=0.5, linecolor="white",
    )
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title("Confusion Matrix")

    plt.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved %s", output_path)


def run(checkpoint_path, test_path, config_path):
    from src.model.classifier import ABSAClassifier

    with open(config_path) as f:
        config = yaml.safe_load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # load model
    model = ABSAClassifier(config["training"]["model_name"]).to(device)
    state_dict = torch.load(checkpoint_path / "model.pt", map_location=device)
    model.load_state_dict(state_dict)
    tokenizer = AutoTokenizer.from_pretrained(checkpoint_path)

    # load test data (human-labeled only)
    samples = load_test_data(test_path)
    logger.info("Loaded %d test samples", len(samples))

    # predict
    preds, probs = predict_batch(model, tokenizer, samples, device)
    true_labels = [s["sentiment"] for s in samples]

    # metrics
    acc = accuracy_score(true_labels, preds)
    macro_f1 = f1_score(true_labels, preds, average="macro", zero_division=0)
    micro_f1 = f1_score(true_labels, preds, average="micro", zero_division=0)

    report = classification_report(
        true_labels, preds, target_names=SENTIMENTS, output_dict=True, zero_division=0
    )

    cm = confusion_matrix(true_labels, preds, labels=SENTIMENTS)

    # per-aspect breakdown
    aspect_metrics = defaultdict(lambda: {"true": [], "pred": []})
    for s, pred in zip(samples, preds):
        aspect_metrics[s["aspect"]]["true"].append(s["sentiment"])
        aspect_metrics[s["aspect"]]["pred"].append(pred)

    per_aspect = {}
    for asp, data in aspect_metrics.items():
        per_aspect[asp] = {
            "f1_macro": round(f1_score(data["true"], data["pred"], average="macro", zero_division=0), 4),
            "accuracy": round(accuracy_score(data["true"], data["pred"]), 4),
            "n": len(data["true"]),
        }

    # save results
    results_dir = PROJECT_ROOT / "results"
    results_dir.mkdir(parents=True, exist_ok=True)

    eval_results = {
        "accuracy": round(acc, 4),
        "macro_f1": round(macro_f1, 4),
        "micro_f1": round(micro_f1, 4),
        "per_class": {k: v for k, v in report.items() if k in SENTIMENTS},
        "per_aspect": per_aspect,
        "confusion_matrix": cm.tolist(),
        "n_samples": len(samples),
    }
    with open(results_dir / "eval_metrics.json", "w") as f:
        json.dump(eval_results, f, indent=2)

    plot_confusion_matrix(cm, results_dir / "confusion_matrix.png")

    # print
    print(f"\n{'='*60}")
    print(f"  Evaluation Results ({len(samples)} samples)")
    print(f"{'='*60}")
    print(f"  Accuracy:  {acc:.4f}")
    print(f"  Macro-F1:  {macro_f1:.4f}")
    print(f"  Micro-F1:  {micro_f1:.4f}")

    print(f"\n  Per-class F1:")
    for sent in SENTIMENTS:
        if sent in report:
            r = report[sent]
            print(f"    {sent:<12s}  P={r['precision']:.3f}  R={r['recall']:.3f}  F1={r['f1-score']:.3f}  n={int(r['support'])}")

    print(f"\n  Per-aspect Macro-F1:")
    for asp, info in sorted(per_aspect.items()):
        print(f"    {asp:<30s}  F1={info['f1_macro']:.4f}  n={info['n']}")

    print(f"{'='*60}\n")

    # MLflow
    try:
        import mlflow
        mlflow.set_tracking_uri(str(PROJECT_ROOT / config["mlflow"]["tracking_uri"]))
        mlflow.set_experiment(config["mlflow"]["experiment_name"])
        with mlflow.start_run(run_name="evaluation"):
            mlflow.log_metrics({
                "eval/accuracy": acc,
                "eval/macro_f1": macro_f1,
                "eval/micro_f1": micro_f1,
            })
            mlflow.log_artifact(str(results_dir / "eval_metrics.json"))
            mlflow.log_artifact(str(results_dir / "confusion_matrix.png"))
    except Exception as e:
        logger.warning("MLflow logging failed: %s", e)


def main():
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--test-data", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=PROJECT_ROOT / "configs" / "config.yaml")
    args = parser.parse_args()
    run(args.checkpoint, args.test_data, args.config)


if __name__ == "__main__":
    main()
