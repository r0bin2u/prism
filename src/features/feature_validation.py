from __future__ import annotations

import argparse
import json
import logging
from collections import defaultdict
from pathlib import Path

import numpy as np
import yaml
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, accuracy_score
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler

from src.features.aspect_features import (
    compute_product_features,
    compute_user_preference,
    compute_cross_features,
)

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[2]

SENTIMENTS = ["positive", "negative", "neutral"]


# -- Batch inference with DeBERTa --

def batch_predict(model, tokenizer, texts, aspects, device, batch_size=64):
    """Run DeBERTa on (text, aspect) pairs in batches. Returns list of
    {"aspect": str, "sentiment": str, "confidence": float}."""
    import torch

    model.eval()
    sentiment_names = ["positive", "negative", "neutral"]
    results = []

    for start in range(0, len(texts), batch_size):
        batch_texts = texts[start:start + batch_size]
        batch_aspects = aspects[start:start + batch_size]

        enc = tokenizer(
            batch_texts, batch_aspects,
            truncation=True, max_length=256,
            padding=True, return_tensors="pt",
        )
        input_ids = enc["input_ids"].to(device)
        attention_mask = enc["attention_mask"].to(device)

        with torch.no_grad():
            logits = model(input_ids, attention_mask)
            probs = torch.softmax(logits, dim=-1).cpu().numpy()

        for i in range(len(batch_texts)):
            idx = probs[i].argmax()
            results.append({
                "aspect": batch_aspects[i],
                "sentiment": sentiment_names[idx],
                "confidence": float(probs[i][idx]),
            })

    return results


def run_inference_on_reviews(reviews, aspect_list, model, tokenizer, device):
    """For each review, predict all aspects. Returns reviews enriched with 'aspects' field."""
    all_texts = []
    all_aspects = []
    review_indices = []

    for i, review in enumerate(reviews):
        for asp in aspect_list:
            all_texts.append(review["text"])
            all_aspects.append(asp)
            review_indices.append(i)

    preds = batch_predict(model, tokenizer, all_texts, all_aspects, device)

    # group by review
    enriched = []
    review_preds = defaultdict(list)
    for idx, pred in zip(review_indices, preds):
        review_preds[idx].append(pred)

    for i, review in enumerate(reviews):
        r = dict(review)
        r["aspects"] = review_preds.get(i, [])
        enriched.append(r)

    return enriched


# -- Data preparation --

def prepare_experiment_data(reviews, aspect_list, model, tokenizer, device):
    """Group reviews by (user, product), build features, time-split."""

    # run DeBERTa on all reviews
    logger.info("Running DeBERTa inference on %d reviews...", len(reviews))
    enriched = run_inference_on_reviews(reviews, aspect_list, model, tokenizer, device)

    # group by product and user
    product_reviews = defaultdict(list)
    user_reviews = defaultdict(list)
    interactions = []  # (user_id, product_id, rating, timestamp, review_idx)

    for i, r in enumerate(enriched):
        uid = r.get("user_id", "unknown")
        pid = r.get("product_id", "unknown")
        rating = r.get("rating", 3.0)
        ts = r.get("timestamp", 0)

        product_reviews[pid].append(r)
        user_reviews[uid].append(r)
        interactions.append((uid, pid, rating, ts, i))

    # sort by timestamp for time-based split
    interactions.sort(key=lambda x: x[3])

    # time split: 80% train, 20% test
    split_idx = int(len(interactions) * 0.8)
    train_interactions = interactions[:split_idx]
    test_interactions = interactions[split_idx:]

    # for training: only use reviews before the split point
    split_ts = interactions[split_idx][3] if split_idx < len(interactions) else float("inf")
    train_product_reviews = defaultdict(list)
    train_user_reviews = defaultdict(list)
    for uid, pid, rating, ts, idx in train_interactions:
        train_product_reviews[pid].append(enriched[idx])
        train_user_reviews[uid].append(enriched[idx])

    return {
        "enriched": enriched,
        "train_interactions": train_interactions,
        "test_interactions": test_interactions,
        "train_product_reviews": train_product_reviews,
        "train_user_reviews": train_user_reviews,
        "product_reviews": product_reviews,
        "aspect_list": aspect_list,
    }


def build_features(interactions, product_reviews_map, user_reviews_map, aspect_list, mode="A"):
    """Build feature matrix for a set of interactions.

    mode A: baseline only (avg_rating, review_count)
    mode B: baseline + product aspect features
    mode C: baseline + product aspect + cross features
    """
    X = []
    y = []

    for uid, pid, rating, ts, idx in interactions:
        label = 1 if rating >= 4.0 else 0

        p_reviews = product_reviews_map.get(pid, [])

        # baseline features
        if p_reviews:
            ratings = [r.get("rating", 3.0) for r in p_reviews]
            avg_rating = sum(ratings) / len(ratings)
            review_count = len(p_reviews)
        else:
            avg_rating = 3.0
            review_count = 0

        baseline = [avg_rating, review_count]

        if mode == "A":
            X.append(baseline)
        elif mode == "B":
            product_feats = compute_product_features(p_reviews, aspect_list)
            aspect_values = []
            for asp in aspect_list:
                for s in SENTIMENTS:
                    aspect_values.append(product_feats[f"{asp}_{s}_ratio"])
                aspect_values.append(product_feats[f"{asp}_review_count"])
            X.append(baseline + aspect_values)
        elif mode == "C":
            product_feats = compute_product_features(p_reviews, aspect_list)
            aspect_values = []
            for asp in aspect_list:
                for s in SENTIMENTS:
                    aspect_values.append(product_feats[f"{asp}_{s}_ratio"])
                aspect_values.append(product_feats[f"{asp}_review_count"])

            u_reviews = user_reviews_map.get(uid, [])
            user_pref = compute_user_preference(u_reviews, aspect_list)
            cross = compute_cross_features(user_pref, product_feats)
            X.append(baseline + aspect_values + cross)

        y.append(label)

    return np.array(X, dtype=np.float32), np.array(y, dtype=np.int32)


# -- Experiment runner --

def run_experiment(X_train, y_train, X_test, y_test, model_type="lr"):
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    if model_type == "lr":
        clf = LogisticRegression(max_iter=1000, random_state=42)
    elif model_type == "mlp":
        clf = MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=500, random_state=42)
    else:
        raise ValueError(f"Unknown model type: {model_type}")

    clf.fit(X_train_s, y_train)

    y_pred = clf.predict(X_test_s)
    y_prob = clf.predict_proba(X_test_s)[:, 1]

    auc = roc_auc_score(y_test, y_prob)
    acc = accuracy_score(y_test, y_pred)

    return {"auc": round(auc, 4), "accuracy": round(acc, 4), "model": clf, "scaler": scaler}


def cold_start_analysis(test_interactions, product_reviews_map, user_reviews_map,
                        aspect_list, model_A, scaler_A, model_B, scaler_B):
    """Split test set by product review count, compute AUC per bucket."""
    buckets = {"cold (1-5)": [], "warm (6-20)": [], "hot (>20)": []}

    for uid, pid, rating, ts, idx in test_interactions:
        n = len(product_reviews_map.get(pid, []))
        if n <= 5:
            buckets["cold (1-5)"].append((uid, pid, rating, ts, idx))
        elif n <= 20:
            buckets["warm (6-20)"].append((uid, pid, rating, ts, idx))
        else:
            buckets["hot (>20)"].append((uid, pid, rating, ts, idx))

    results = {}
    for bucket_name, bucket_interactions in buckets.items():
        if len(bucket_interactions) < 10:
            results[bucket_name] = {"n": len(bucket_interactions), "auc_A": None, "auc_B": None}
            continue

        X_A, y_A = build_features(bucket_interactions, product_reviews_map, user_reviews_map,
                                  aspect_list, mode="A")
        X_B, y_B = build_features(bucket_interactions, product_reviews_map, user_reviews_map,
                                  aspect_list, mode="B")

        # need both classes in test
        if len(set(y_A)) < 2:
            results[bucket_name] = {"n": len(bucket_interactions), "auc_A": None, "auc_B": None}
            continue

        prob_A = model_A.predict_proba(scaler_A.transform(X_A))[:, 1]
        prob_B = model_B.predict_proba(scaler_B.transform(X_B))[:, 1]

        auc_A = roc_auc_score(y_A, prob_A)
        auc_B = roc_auc_score(y_B, prob_B)

        results[bucket_name] = {
            "n": len(bucket_interactions),
            "auc_A": round(auc_A, 4),
            "auc_B": round(auc_B, 4),
            "lift": round(auc_B - auc_A, 4),
        }

    return results


def learning_curve_by_review_count(test_interactions, product_reviews_map, user_reviews_map,
                                   aspect_list, model_A, scaler_A, model_B, scaler_B):
    """Compute AUC at increasing review-count thresholds.
    For threshold K, only include test samples where the product has <= K reviews."""
    thresholds = [1, 2, 3, 5, 8, 12, 20, 30, 50, 100]
    curve = []

    for k in thresholds:
        subset = [(uid, pid, r, ts, idx) for uid, pid, r, ts, idx in test_interactions
                  if len(product_reviews_map.get(pid, [])) <= k]

        if len(subset) < 30:
            continue

        X_A, y_A = build_features(subset, product_reviews_map, user_reviews_map, aspect_list, "A")
        X_B, y_B = build_features(subset, product_reviews_map, user_reviews_map, aspect_list, "B")

        if len(set(y_A)) < 2:
            continue

        prob_A = model_A.predict_proba(scaler_A.transform(X_A))[:, 1]
        prob_B = model_B.predict_proba(scaler_B.transform(X_B))[:, 1]

        curve.append({
            "max_reviews": k,
            "n": len(subset),
            "auc_A": round(roc_auc_score(y_A, prob_A), 4),
            "auc_B": round(roc_auc_score(y_B, prob_B), 4),
        })

    return curve


def simulated_cold_start(test_interactions, enriched_reviews, product_reviews_map,
                         user_reviews_map, aspect_list, model_A, scaler_A, model_B, scaler_B,
                         min_product_reviews=50):
    """For products with many reviews, artificially truncate to K reviews and measure AUC.
    This controls for product differences — same products, different info available."""
    truncation_points = [1, 3, 5, 10, 20, 50]

    # find test interactions where product has >= min_product_reviews
    eligible = [(uid, pid, r, ts, idx) for uid, pid, r, ts, idx in test_interactions
                if len(product_reviews_map.get(pid, [])) >= min_product_reviews]

    if len(eligible) < 30:
        return []

    curve = []
    for k in truncation_points:
        # build truncated product review map: only keep first K reviews per product (by timestamp)
        truncated_map = {}
        for pid, reviews in product_reviews_map.items():
            sorted_reviews = sorted(reviews, key=lambda r: r.get("timestamp", 0))
            truncated_map[pid] = sorted_reviews[:k]

        X_A, y_A = build_features(eligible, truncated_map, user_reviews_map, aspect_list, "A")
        X_B, y_B = build_features(eligible, truncated_map, user_reviews_map, aspect_list, "B")

        if len(set(y_A)) < 2:
            continue

        prob_A = model_A.predict_proba(scaler_A.transform(X_A))[:, 1]
        prob_B = model_B.predict_proba(scaler_B.transform(X_B))[:, 1]

        curve.append({
            "k_reviews": k,
            "n": len(eligible),
            "auc_A": round(roc_auc_score(y_A, prob_A), 4),
            "auc_B": round(roc_auc_score(y_B, prob_B), 4),
            "lift": round(roc_auc_score(y_B, prob_B) - roc_auc_score(y_A, prob_A), 4),
        })

    # also run with full reviews for comparison
    X_A_full, y_A_full = build_features(eligible, product_reviews_map, user_reviews_map, aspect_list, "A")
    X_B_full, y_B_full = build_features(eligible, product_reviews_map, user_reviews_map, aspect_list, "B")
    if len(set(y_A_full)) >= 2:
        curve.append({
            "k_reviews": "all",
            "n": len(eligible),
            "auc_A": round(roc_auc_score(y_A_full, model_A.predict_proba(scaler_A.transform(X_A_full))[:, 1]), 4),
            "auc_B": round(roc_auc_score(y_B_full, model_B.predict_proba(scaler_B.transform(X_B_full))[:, 1]), 4),
            "lift": round(
                roc_auc_score(y_B_full, model_B.predict_proba(scaler_B.transform(X_B_full))[:, 1])
                - roc_auc_score(y_A_full, model_A.predict_proba(scaler_A.transform(X_A_full))[:, 1]), 4),
        })

    return curve


def feature_importance(model_lr, aspect_list):
    """Extract LR coefficients as feature importance."""
    if not hasattr(model_lr, "coef_"):
        return {}

    coefs = model_lr.coef_[0]
    # first 2 features are baseline (avg_rating, review_count)
    # then per aspect: pos_ratio, neg_ratio, neu_ratio, count = 4 features
    importance = {}
    offset = 2
    for asp in aspect_list:
        for j, s in enumerate(SENTIMENTS):
            feat_name = f"{asp}_{s}_ratio"
            if offset + j < len(coefs):
                importance[feat_name] = round(float(coefs[offset + j]), 4)
        count_name = f"{asp}_review_count"
        if offset + len(SENTIMENTS) < len(coefs):
            importance[count_name] = round(float(coefs[offset + len(SENTIMENTS)]), 4)
        offset += len(SENTIMENTS) + 1

    # sort by absolute value
    importance = dict(sorted(importance.items(), key=lambda x: -abs(x[1])))
    return importance


# -- Visualization (Morandi palette) --

MORANDI = {
    "sage":     "#A2A88F",
    "dusty_rose": "#C9A9A6",
    "slate":    "#8E9AAF",
    "clay":     "#C4A882",
    "lavender": "#B8B8D1",
    "moss":     "#7D8570",
    "blush":    "#D4B5B0",
    "stone":    "#9B998D",
    "mauve":    "#BFA5B8",
    "sand":     "#C7BEA2",
}


def _setup_plot_style():
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
        "grid.alpha": 0.7,
        "font.size": 11,
        "axes.titlesize": 13,
        "axes.titleweight": "medium",
    })


def plot_experiment_comparison(results, output_dir):
    import matplotlib.pyplot as plt
    _setup_plot_style()

    names = list(results.keys())
    lr_aucs = [results[n]["lr"]["auc"] for n in names]
    mlp_aucs = [results[n]["mlp"]["auc"] for n in names]

    x = np.arange(len(names))
    width = 0.32

    fig, ax = plt.subplots(figsize=(8, 5))
    bars1 = ax.bar(x - width/2, lr_aucs, width, label="Logistic Regression",
                   color=MORANDI["sage"], edgecolor="white", linewidth=0.8)
    bars2 = ax.bar(x + width/2, mlp_aucs, width, label="MLP",
                   color=MORANDI["dusty_rose"], edgecolor="white", linewidth=0.8)

    # value labels on bars
    for bars in [bars1, bars2]:
        for bar in bars:
            h = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2, h + 0.003,
                    f"{h:.3f}", ha="center", va="bottom", fontsize=9, color="#4A4A4A")

    ax.set_ylabel("AUC-ROC")
    ax.set_title("Feature Ablation: Aspect Features Improve Prediction")
    ax.set_xticks(x)
    short_names = ["Baseline", "+ Aspect", "+ Aspect + Cross"]
    ax.set_xticklabels(short_names[:len(names)])
    ax.legend(frameon=False)
    ax.set_ylim(min(lr_aucs + mlp_aucs) - 0.05, max(lr_aucs + mlp_aucs) + 0.05)
    ax.grid(axis="y", linestyle="--")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    plt.tight_layout()
    path = Path(output_dir) / "experiment_comparison.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved %s", path)


def plot_cold_start(cold_results, output_dir):
    import matplotlib.pyplot as plt
    _setup_plot_style()

    buckets = []
    auc_a = []
    auc_b = []
    for name, info in cold_results.items():
        if info.get("auc_A") is not None:
            buckets.append(name)
            auc_a.append(info["auc_A"])
            auc_b.append(info["auc_B"])

    if not buckets:
        return

    x = np.arange(len(buckets))
    width = 0.32

    fig, ax = plt.subplots(figsize=(8, 5))
    bars1 = ax.bar(x - width/2, auc_a, width, label="Baseline (no aspect)",
                   color=MORANDI["stone"], edgecolor="white", linewidth=0.8)
    bars2 = ax.bar(x + width/2, auc_b, width, label="+ Aspect features",
                   color=MORANDI["slate"], edgecolor="white", linewidth=0.8)

    # lift annotations
    for i in range(len(buckets)):
        lift = auc_b[i] - auc_a[i]
        mid_x = x[i] + width/2
        ax.annotate(f"+{lift:.3f}", xy=(mid_x, auc_b[i]),
                    xytext=(mid_x + 0.15, auc_b[i] + 0.015),
                    fontsize=9, color=MORANDI["moss"], fontweight="bold",
                    arrowprops=dict(arrowstyle="-", color=MORANDI["moss"], lw=0.8))

    ax.set_ylabel("AUC-ROC")
    ax.set_title("Cold-Start Analysis: Aspect Features Help Most with Few Reviews")
    ax.set_xticks(x)
    ax.set_xticklabels(buckets)
    ax.legend(frameon=False)
    ax.set_ylim(min(auc_a + auc_b) - 0.05, max(auc_a + auc_b) + 0.06)
    ax.grid(axis="y", linestyle="--")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    plt.tight_layout()
    path = Path(output_dir) / "cold_start_analysis.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved %s", path)


def plot_feature_importance(importance, output_dir, top_k=10):
    import matplotlib.pyplot as plt
    _setup_plot_style()

    items = list(importance.items())[:top_k]
    items.reverse()  # so highest importance is at top of horizontal bar chart

    names = [name for name, _ in items]
    values = [val for _, val in items]
    colors = [MORANDI["sage"] if v >= 0 else MORANDI["dusty_rose"] for v in values]

    fig, ax = plt.subplots(figsize=(9, 5))
    bars = ax.barh(names, values, color=colors, edgecolor="white", linewidth=0.8, height=0.6)

    # value labels
    for bar, val in zip(bars, values):
        offset = 0.005 if val >= 0 else -0.005
        ha = "left" if val >= 0 else "right"
        ax.text(val + offset, bar.get_y() + bar.get_height()/2,
                f"{val:+.4f}", ha=ha, va="center", fontsize=9, color="#4A4A4A")

    ax.axvline(x=0, color=MORANDI["stone"], linewidth=0.8)
    ax.set_xlabel("LR Coefficient (standardized)")
    ax.set_title("Top Feature Importance: Which Aspects Drive Predictions")
    ax.grid(axis="x", linestyle="--")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    plt.tight_layout()
    path = Path(output_dir) / "feature_importance.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved %s", path)


def plot_learning_curve(curve_data, output_dir):
    import matplotlib.pyplot as plt
    _setup_plot_style()

    if not curve_data:
        return

    ks = [d["max_reviews"] for d in curve_data]
    auc_a = [d["auc_A"] for d in curve_data]
    auc_b = [d["auc_B"] for d in curve_data]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(ks, auc_a, "o-", color=MORANDI["stone"], label="Baseline", linewidth=2, markersize=6)
    ax.plot(ks, auc_b, "s-", color=MORANDI["sage"], label="+ Aspect features", linewidth=2, markersize=6)

    # shade the gap
    ax.fill_between(ks, auc_a, auc_b, alpha=0.15, color=MORANDI["sage"])

    ax.set_xlabel("Max product review count (≤ K)")
    ax.set_ylabel("AUC-ROC")
    ax.set_title("Learning Curve: Aspect Feature Advantage Shrinks with More Reviews")
    ax.legend(frameon=False)
    ax.grid(axis="y", linestyle="--")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    plt.tight_layout()
    path = Path(output_dir) / "learning_curve.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved %s", path)


def plot_simulated_cold_start(sim_data, output_dir):
    import matplotlib.pyplot as plt
    _setup_plot_style()

    if not sim_data:
        return

    # separate numeric and "all" points
    numeric = [d for d in sim_data if isinstance(d["k_reviews"], int)]
    full = [d for d in sim_data if d["k_reviews"] == "all"]

    if not numeric:
        return

    ks = [d["k_reviews"] for d in numeric]
    auc_a = [d["auc_A"] for d in numeric]
    auc_b = [d["auc_B"] for d in numeric]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(ks, auc_a, "o-", color=MORANDI["stone"], label="Baseline", linewidth=2, markersize=6)
    ax.plot(ks, auc_b, "s-", color=MORANDI["slate"], label="+ Aspect features", linewidth=2, markersize=6)

    # show "all reviews" as dashed horizontal lines
    if full:
        ax.axhline(y=full[0]["auc_A"], color=MORANDI["stone"], linestyle="--", alpha=0.5)
        ax.axhline(y=full[0]["auc_B"], color=MORANDI["slate"], linestyle="--", alpha=0.5)
        ax.text(ks[-1] * 1.05, full[0]["auc_B"] + 0.003, "all reviews", fontsize=8, color=MORANDI["slate"])

    ax.fill_between(ks, auc_a, auc_b, alpha=0.15, color=MORANDI["slate"])

    ax.set_xlabel("Number of reviews available per product (truncated)")
    ax.set_ylabel("AUC-ROC")
    ax.set_title("Simulated Cold-Start: Same Products, Varying Information")
    ax.legend(frameon=False)
    ax.grid(axis="y", linestyle="--")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    plt.tight_layout()
    path = Path(output_dir) / "simulated_cold_start.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved %s", path)


def generate_all_plots(results, cold_results, importance, output_dir,
                       learning_curve_data=None, simulated_data=None):
    plot_experiment_comparison(results, output_dir)
    plot_cold_start(cold_results, output_dir)
    plot_feature_importance(importance, output_dir)
    if learning_curve_data:
        plot_learning_curve(learning_curve_data, output_dir)
    if simulated_data:
        plot_simulated_cold_start(simulated_data, output_dir)


# -- Main --

def run(amazon_data_path, checkpoint_path, config_path, output_dir, max_reviews=50000):
    import torch
    from src.model.classifier import ABSAClassifier
    from transformers import AutoTokenizer

    with open(config_path) as f:
        config = yaml.safe_load(f)

    aspect_list = config["aspects"]["general"]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # load model
    model = ABSAClassifier(config["training"]["model_name"]).to(device)
    state_dict = torch.load(checkpoint_path / "model.pt", map_location=device)
    model.load_state_dict(state_dict)
    tokenizer = AutoTokenizer.from_pretrained(checkpoint_path)
    logger.info("Loaded model from %s", checkpoint_path)

    # load amazon reviews
    reviews = []
    with open(amazon_data_path) as f:
        for line in f:
            if not line.strip():
                continue
            reviews.append(json.loads(line))
            if len(reviews) >= max_reviews:
                break
    logger.info("Loaded %d reviews", len(reviews))

    # prepare data
    data = prepare_experiment_data(reviews, aspect_list, model, tokenizer, device)

    train_int = data["train_interactions"]
    test_int = data["test_interactions"]
    train_pr = data["train_product_reviews"]
    train_ur = data["train_user_reviews"]

    logger.info("Train: %d, Test: %d", len(train_int), len(test_int))

    # run three experiments
    results = {}
    models = {}
    scalers = {}

    for mode, name in [("A", "baseline"), ("B", "baseline + aspect"), ("C", "baseline + aspect + cross")]:
        X_train, y_train = build_features(train_int, train_pr, train_ur, aspect_list, mode)
        X_test, y_test = build_features(test_int, train_pr, train_ur, aspect_list, mode)

        lr_result = run_experiment(X_train, y_train, X_test, y_test, "lr")
        mlp_result = run_experiment(X_train, y_train, X_test, y_test, "mlp")

        results[name] = {
            "mode": mode,
            "lr": {"auc": lr_result["auc"], "accuracy": lr_result["accuracy"]},
            "mlp": {"auc": mlp_result["auc"], "accuracy": mlp_result["accuracy"]},
            "n_features": X_train.shape[1],
            "train_size": len(y_train),
            "test_size": len(y_test),
            "positive_ratio": float(y_test.mean()),
        }
        models[mode] = lr_result["model"]
        scalers[mode] = lr_result["scaler"]

    # cold start analysis
    cold_results = cold_start_analysis(
        test_int, train_pr, train_ur, aspect_list,
        models["A"], scalers["A"], models["B"], scalers["B"],
    )

    # learning curve: AUC vs max review count
    lc_data = learning_curve_by_review_count(
        test_int, train_pr, train_ur, aspect_list,
        models["A"], scalers["A"], models["B"], scalers["B"],
    )

    # simulated cold start: truncate hot products
    sim_data = simulated_cold_start(
        test_int, data["enriched"], train_pr, train_ur, aspect_list,
        models["A"], scalers["A"], models["B"], scalers["B"],
    )

    # feature importance from LR model B
    importance = feature_importance(models["B"], aspect_list)

    # save results
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    full_results = {
        "experiments": results,
        "cold_start": cold_results,
        "learning_curve": lc_data,
        "simulated_cold_start": sim_data,
        "feature_importance": importance,
    }
    with open(output_dir / "feature_validation.json", "w") as f:
        json.dump(full_results, f, indent=2)

    # generate plots
    generate_all_plots(results, cold_results, importance, output_dir, lc_data, sim_data)

    # print
    print(f"\n{'='*65}")
    print(f"  Feature Validation Results")
    print(f"{'='*65}")

    print(f"\n  Experiment Comparison (AUC-ROC):")
    print(f"  {'Experiment':<30s} {'LR':>8s} {'MLP':>8s} {'Features':>10s}")
    print(f"  {'-'*58}")
    for name, res in results.items():
        print(f"  {name:<30s} {res['lr']['auc']:>8.4f} {res['mlp']['auc']:>8.4f} {res['n_features']:>10d}")

    print(f"\n  Cold Start Analysis (LR AUC):")
    print(f"  {'Bucket':<20s} {'n':>6s} {'Baseline':>10s} {'+ Aspect':>10s} {'Lift':>8s}")
    print(f"  {'-'*56}")
    for bucket, info in cold_results.items():
        if info["auc_A"] is not None:
            print(f"  {bucket:<20s} {info['n']:>6d} {info['auc_A']:>10.4f} {info['auc_B']:>10.4f} {info['lift']:>+8.4f}")
        else:
            print(f"  {bucket:<20s} {info['n']:>6d} {'N/A':>10s} {'N/A':>10s} {'N/A':>8s}")

    if lc_data:
        print(f"\n  Learning Curve (AUC vs max review count):")
        print(f"  {'≤K reviews':>12s} {'n':>6s} {'Baseline':>10s} {'+ Aspect':>10s} {'Gap':>8s}")
        print(f"  {'-'*48}")
        for d in lc_data:
            gap = d["auc_B"] - d["auc_A"]
            print(f"  {'≤'+str(d['max_reviews']):>12s} {d['n']:>6d} {d['auc_A']:>10.4f} {d['auc_B']:>10.4f} {gap:>+8.4f}")

    if sim_data:
        print(f"\n  Simulated Cold-Start (hot products truncated):")
        print(f"  {'K reviews':>12s} {'n':>6s} {'Baseline':>10s} {'+ Aspect':>10s} {'Lift':>8s}")
        print(f"  {'-'*48}")
        for d in sim_data:
            print(f"  {str(d['k_reviews']):>12s} {d['n']:>6d} {d['auc_A']:>10.4f} {d['auc_B']:>10.4f} {d['lift']:>+8.4f}")

    print(f"\n  Top 10 Feature Importance (LR coefficients):")
    for i, (feat, coef) in enumerate(list(importance.items())[:10]):
        print(f"  {i+1:>3d}. {feat:<35s} {coef:>+.4f}")

    print(f"{'='*65}\n")

    # MLflow logging
    try:
        import mlflow
        mlflow.set_tracking_uri(str(PROJECT_ROOT / config["mlflow"]["tracking_uri"]))
        mlflow.set_experiment(config["mlflow"]["experiment_name"])
        with mlflow.start_run(run_name="feature-validation"):
            for name, res in results.items():
                mlflow.log_metrics({
                    f"fv/{name}/lr_auc": res["lr"]["auc"],
                    f"fv/{name}/mlp_auc": res["mlp"]["auc"],
                })
            for bucket, info in cold_results.items():
                if info["auc_A"] is not None:
                    tag = bucket.split("(")[0].strip().replace(" ", "_")
                    mlflow.log_metrics({
                        f"fv/cold/{tag}/baseline_auc": info["auc_A"],
                        f"fv/cold/{tag}/aspect_auc": info["auc_B"],
                        f"fv/cold/{tag}/lift": info["lift"],
                    })
            mlflow.log_artifact(str(output_dir / "feature_validation.json"))
    except Exception as e:
        logger.warning("MLflow logging failed: %s", e)


def main():
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
    parser = argparse.ArgumentParser()
    parser.add_argument("--amazon-data", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=PROJECT_ROOT / "configs" / "config.yaml")
    parser.add_argument("--output", type=Path, default=PROJECT_ROOT / "results")
    parser.add_argument("--max-reviews", type=int, default=50000)
    args = parser.parse_args()
    run(args.amazon_data, args.checkpoint, args.config, args.output, args.max_reviews)


if __name__ == "__main__":
    main()
