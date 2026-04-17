"""Pairwise ranking variant of feature validation.

Instead of predicting whether a user rates a product >= 4 (pointwise binary),
this script evaluates whether the model can rank products within the same user:
given (user A, product X, product Y) where rating_X != rating_Y, predict which
one the user prefers.

Reuses DeBERTa inference and feature construction from feature_validation.py.
"""

from __future__ import annotations

import argparse
import json
import logging
import random
from collections import defaultdict
from pathlib import Path

import numpy as np
import yaml
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler

from src.features.aspect_features import (
    SENTIMENTS,
    compute_cross_features,
    compute_product_features,
    compute_user_preference,
)
from src.features.feature_validation import (
    prepare_experiment_data,
    run_inference_on_reviews,
)

logger = logging.getLogger(__name__)
PROJECT_ROOT = Path(__file__).resolve().parents[2]


def product_feature_vector(pid, product_reviews_map, aspect_list, mode, user_pref=None):
    """Build a single product's feature vector in the given mode."""
    p_reviews = product_reviews_map.get(pid, [])

    if p_reviews:
        ratings = [r.get("rating", 3.0) for r in p_reviews]
        avg_rating = sum(ratings) / len(ratings)
        review_count = len(p_reviews)
    else:
        avg_rating = 3.0
        review_count = 0

    baseline = [avg_rating, review_count]

    if mode == "A":
        return baseline

    product_feats = compute_product_features(p_reviews, aspect_list)
    aspect_values = []
    for asp in aspect_list:
        for s in SENTIMENTS:
            aspect_values.append(product_feats[f"{asp}_{s}_ratio"])
        aspect_values.append(product_feats[f"{asp}_review_count"])

    if mode == "B":
        return baseline + aspect_values

    # mode C: add cross features (requires user preference)
    cross = compute_cross_features(user_pref, product_feats) if user_pref else [1 / 3] * 9
    return baseline + aspect_values + cross


def build_pairs(interactions, product_reviews_map, user_reviews_map, aspect_list, mode, max_pairs_per_user=10, seed=42):
    """For each user with >=2 reviews, construct ordered pairs (X, Y) with rating_X != rating_Y.
    Features: f(X) - f(Y). Label: 1 if rating_X > rating_Y else 0."""
    rng = random.Random(seed)

    # group by user
    by_user = defaultdict(list)
    for uid, pid, rating, ts, idx in interactions:
        by_user[uid].append((pid, rating))

    X_feats = []
    y = []

    for uid, items in by_user.items():
        if len(items) < 2:
            continue

        # precompute user preference once per user (for mode C)
        u_reviews = user_reviews_map.get(uid, [])
        user_pref = compute_user_preference(u_reviews, aspect_list) if mode == "C" else None

        # precompute feature vectors per product
        feat_cache = {}
        for pid, _ in items:
            if pid not in feat_cache:
                feat_cache[pid] = product_feature_vector(pid, product_reviews_map, aspect_list, mode, user_pref)

        # generate all pairs, shuffle, take top-K to cap dominant-user bias
        pairs = []
        for i in range(len(items)):
            for j in range(len(items)):
                if i == j:
                    continue
                pid_i, r_i = items[i]
                pid_j, r_j = items[j]
                if r_i == r_j:
                    continue
                pairs.append((pid_i, pid_j, r_i, r_j))

        if len(pairs) > max_pairs_per_user:
            pairs = rng.sample(pairs, max_pairs_per_user)

        for pid_i, pid_j, r_i, r_j in pairs:
            f_i = feat_cache[pid_i]
            f_j = feat_cache[pid_j]
            diff = [a - b for a, b in zip(f_i, f_j)]
            X_feats.append(diff)
            y.append(1 if r_i > r_j else 0)

    return np.array(X_feats, dtype=np.float32), np.array(y, dtype=np.int32)


def run_pairwise_experiment(X_train, y_train, X_test, y_test, model_type="lr"):
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    if model_type == "lr":
        clf = LogisticRegression(max_iter=1000, C=1.0)
    else:
        clf = MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=200, random_state=42)

    clf.fit(X_train_s, y_train)
    prob = clf.predict_proba(X_test_s)[:, 1]
    auc = roc_auc_score(y_test, prob)
    acc = clf.score(X_test_s, y_test)
    return {"auc": auc, "accuracy": acc, "n_features": X_train.shape[1]}


def run(amazon_data_path, checkpoint_path, config_path, max_reviews=50000):
    import torch
    from src.model.classifier import ABSAClassifier
    from transformers import AutoTokenizer

    with open(config_path) as f:
        config = yaml.safe_load(f)

    aspect_list = config["aspects"].get("restaurant", config["aspects"]["general"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = ABSAClassifier(config["training"]["model_name"]).to(device)
    state_dict = torch.load(checkpoint_path / "model.pt", map_location=device)
    model.load_state_dict(state_dict)
    tokenizer = AutoTokenizer.from_pretrained(checkpoint_path)

    reviews = []
    with open(amazon_data_path) as f:
        for line in f:
            if not line.strip():
                continue
            reviews.append(json.loads(line))
            if len(reviews) >= max_reviews:
                break
    logger.info("Loaded %d reviews", len(reviews))

    # reuse inference + time split from feature_validation
    data = prepare_experiment_data(reviews, aspect_list, model, tokenizer, device)

    # build pairs for train / test
    print(f"\n{'='*65}")
    print(f"  Pairwise Ranking Experiment")
    print(f"{'='*65}")
    print(f"{'Experiment':<25} {'LR AUC':>10} {'MLP AUC':>10} {'Pairs':>10}")
    print("-" * 65)

    results = {}
    for mode, name in [("A", "baseline"), ("B", "+ aspect"), ("C", "+ aspect + cross")]:
        X_train, y_train = build_pairs(
            data["train_interactions"],
            data["train_product_reviews"],
            data["train_user_reviews"],
            aspect_list, mode,
        )
        X_test, y_test = build_pairs(
            data["test_interactions"],
            data["train_product_reviews"],   # use train-period user/product context
            data["train_user_reviews"],
            aspect_list, mode,
        )

        if len(y_train) < 100 or len(y_test) < 50:
            print(f"  {name:<25} not enough pairs (train={len(y_train)}, test={len(y_test)})")
            continue

        lr = run_pairwise_experiment(X_train, y_train, X_test, y_test, "lr")
        mlp = run_pairwise_experiment(X_train, y_train, X_test, y_test, "mlp")
        results[name] = {"mode": mode, "lr": lr, "mlp": mlp, "train_pairs": len(y_train), "test_pairs": len(y_test)}
        print(f"  {name:<25} {lr['auc']:>10.4f} {mlp['auc']:>10.4f} {len(y_train):>10}")

    print("=" * 65)

    # save
    out_path = PROJECT_ROOT / "results" / "pairwise_results.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    logger.info("Saved %s", out_path)


def main():
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")

    parser = argparse.ArgumentParser()
    parser.add_argument("--amazon-data", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=PROJECT_ROOT / "configs" / "config.yaml")
    parser.add_argument("--max-reviews", type=int, default=50000)
    args = parser.parse_args()

    run(args.amazon_data, args.checkpoint, args.config, args.max_reviews)


if __name__ == "__main__":
    main()
