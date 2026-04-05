from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import numpy as np
import torch
import yaml
from scipy.optimize import minimize_scalar
from transformers import AutoTokenizer

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[2]

SENTIMENTS = ["positive", "negative", "neutral"]


def load_data_as_samples(data_path):
    samples = []
    with open(data_path) as f:
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
                    "label_idx": SENTIMENTS.index(asp["sentiment"]),
                })
    return samples


def get_logits(model, tokenizer, samples, device, batch_size=64):
    model.eval()
    all_logits = []

    for start in range(0, len(samples), batch_size):
        batch = samples[start:start + batch_size]
        texts = [s["text"] for s in batch]
        aspects = [s["aspect"] for s in batch]

        enc = tokenizer(
            texts, aspects,
            truncation=True, max_length=256,
            padding=True, return_tensors="pt",
        )

        with torch.no_grad():
            logits = model(enc["input_ids"].to(device), enc["attention_mask"].to(device))
            all_logits.append(logits.cpu())

    return torch.cat(all_logits, dim=0)


def compute_ece(probs, labels, n_bins=10):
    """Expected Calibration Error. probs: (N, C), labels: (N,) int."""
    confidences = probs.max(axis=1)
    predictions = probs.argmax(axis=1)
    correct = (predictions == labels).astype(float)

    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    total = len(labels)

    bin_details = []
    for i in range(n_bins):
        lo, hi = bin_boundaries[i], bin_boundaries[i + 1]
        mask = (confidences > lo) & (confidences <= hi)
        count = mask.sum()
        if count == 0:
            continue

        avg_conf = confidences[mask].mean()
        avg_acc = correct[mask].mean()
        gap = abs(avg_conf - avg_acc)
        ece += (count / total) * gap

        bin_details.append({
            "bin": f"({lo:.1f}, {hi:.1f}]",
            "count": int(count),
            "avg_confidence": round(float(avg_conf), 4),
            "avg_accuracy": round(float(avg_acc), 4),
            "gap": round(float(gap), 4),
        })

    return float(ece), bin_details


def softmax_np(logits):
    exp = np.exp(logits - logits.max(axis=1, keepdims=True))
    return exp / exp.sum(axis=1, keepdims=True)


def find_optimal_temperature(val_logits, val_labels):
    """Find T that minimizes NLL on validation set."""
    logits_np = val_logits.numpy()

    def nll_at_temperature(T):
        scaled = logits_np / T
        probs = softmax_np(scaled)
        # NLL = -mean(log(prob of correct class))
        correct_probs = probs[np.arange(len(val_labels)), val_labels]
        correct_probs = np.clip(correct_probs, 1e-10, 1.0)
        return -np.log(correct_probs).mean()

    result = minimize_scalar(nll_at_temperature, bounds=(0.1, 10.0), method="bounded")
    return result.x


def plot_reliability_diagram(probs_before, probs_after, labels, output_path, n_bins=10):
    import matplotlib.pyplot as plt

    plt.rcParams.update({
        "figure.facecolor": "#F5F2EE",
        "axes.facecolor": "#F5F2EE",
        "text.color": "#4A4A4A",
        "axes.edgecolor": "#9B998D",
        "font.size": 11,
    })

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    for ax, probs, title in [
        (axes[0], probs_before, "Before Calibration"),
        (axes[1], probs_after, "After Calibration"),
    ]:
        confidences = probs.max(axis=1)
        predictions = probs.argmax(axis=1)
        correct = (predictions == labels).astype(float)

        bin_boundaries = np.linspace(0, 1, n_bins + 1)
        bin_confs = []
        bin_accs = []

        for i in range(n_bins):
            lo, hi = bin_boundaries[i], bin_boundaries[i + 1]
            mask = (confidences > lo) & (confidences <= hi)
            if mask.sum() == 0:
                continue
            bin_confs.append(confidences[mask].mean())
            bin_accs.append(correct[mask].mean())

        ax.bar(bin_confs, bin_accs, width=0.08, alpha=0.7,
               color="#8E9AAF", edgecolor="white", linewidth=0.8, label="Actual accuracy")
        ax.plot([0, 1], [0, 1], "--", color="#9B998D", linewidth=1, label="Perfect calibration")
        ax.set_xlabel("Confidence")
        ax.set_ylabel("Accuracy")
        ax.set_title(title)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.legend(frameon=False, fontsize=9)
        ax.grid(alpha=0.3, linestyle="--")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    plt.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved %s", output_path)


def run(checkpoint_path, val_path, test_path, config_path):
    from src.model.classifier import ABSAClassifier

    with open(config_path) as f:
        config = yaml.safe_load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = ABSAClassifier(config["training"]["model_name"]).to(device)
    state_dict = torch.load(checkpoint_path / "model.pt", map_location=device)
    model.load_state_dict(state_dict)
    tokenizer = AutoTokenizer.from_pretrained(checkpoint_path)

    # get logits for val and test
    val_samples = load_data_as_samples(val_path)
    test_samples = load_data_as_samples(test_path)

    val_logits = get_logits(model, tokenizer, val_samples, device)
    test_logits = get_logits(model, tokenizer, test_samples, device)

    val_labels = np.array([s["label_idx"] for s in val_samples])
    test_labels = np.array([s["label_idx"] for s in test_samples])

    # step 1: diagnose with ECE on test set (before calibration)
    test_logits_np = test_logits.numpy()
    probs_before = softmax_np(test_logits_np)
    ece_before, bins_before = compute_ece(probs_before, test_labels)
    logger.info("ECE before calibration: %.4f", ece_before)

    # step 2: find optimal T on validation set
    optimal_T = find_optimal_temperature(val_logits, val_labels)
    logger.info("Optimal temperature: %.4f", optimal_T)

    # step 3: verify on test set (after calibration)
    probs_after = softmax_np(test_logits_np / optimal_T)
    ece_after, bins_after = compute_ece(probs_after, test_labels)
    logger.info("ECE after calibration: %.4f", ece_after)

    # save calibration T
    cal_path = checkpoint_path / "calibration_T.json"
    cal_data = {
        "temperature": round(optimal_T, 4),
        "ece_before": round(ece_before, 4),
        "ece_after": round(ece_after, 4),
    }
    with open(cal_path, "w") as f:
        json.dump(cal_data, f, indent=2)

    # save reliability diagram
    results_dir = PROJECT_ROOT / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    plot_reliability_diagram(probs_before, probs_after, test_labels,
                            results_dir / "calibration_plot.png")

    # print
    print(f"\n{'='*55}")
    print(f"  Calibration Results")
    print(f"{'='*55}")
    print(f"  Validation samples:  {len(val_samples)}")
    print(f"  Test samples:        {len(test_samples)}")
    print(f"  Optimal T:           {optimal_T:.4f}")
    print(f"  ECE before:          {ece_before:.4f}")
    print(f"  ECE after:           {ece_after:.4f}")
    if ece_before > 0:
        reduction = (ece_before - ece_after) / ece_before * 100
        print(f"  ECE reduction:       {reduction:.1f}%")
    print(f"  Saved T to:          {cal_path}")
    print(f"{'='*55}\n")

    # MLflow
    try:
        import mlflow
        mlflow.set_tracking_uri(str(PROJECT_ROOT / config["mlflow"]["tracking_uri"]))
        mlflow.set_experiment(config["mlflow"]["experiment_name"])
        with mlflow.start_run(run_name="calibration"):
            mlflow.log_metrics({
                "cal/ece_before": ece_before,
                "cal/ece_after": ece_after,
                "cal/temperature": optimal_T,
            })
            mlflow.log_artifact(str(results_dir / "calibration_plot.png"))
            mlflow.log_artifact(str(cal_path))
    except Exception as e:
        logger.warning("MLflow logging failed: %s", e)


def main():
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--val-data", type=Path, required=True)
    parser.add_argument("--test-data", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=PROJECT_ROOT / "configs" / "config.yaml")
    args = parser.parse_args()
    run(args.checkpoint, args.val_data, args.test_data, args.config)


if __name__ == "__main__":
    main()
