from __future__ import annotations

import argparse
import json
import logging
import math
import random
from pathlib import Path

import torch
import yaml
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def compute_entropy(probs):
    """Entropy of a probability distribution. Higher = more uncertain."""
    entropy = 0.0
    for p in probs:
        if p > 0:
            entropy -= p * math.log(p)
    return entropy


def score_unlabeled(model, tokenizer, reviews, aspect_list, device, max_length=256, batch_size=64):
    """Run model on all (review, aspect) pairs, return entropy scores per review.

    Returns list of (review_id, avg_entropy) sorted by entropy descending.
    """
    model.eval()

    # build all (review_idx, aspect) pairs
    pairs = []
    for i, review in enumerate(reviews):
        for aspect in aspect_list:
            pairs.append((i, review["text"], aspect))

    # batch inference
    review_entropies = {}  # review_idx -> list of entropies

    for start in range(0, len(pairs), batch_size):
        batch_pairs = pairs[start:start + batch_size]
        texts = [p[1] for p in batch_pairs]
        aspects = [p[2] for p in batch_pairs]
        idxs = [p[0] for p in batch_pairs]

        encodings = tokenizer(
            texts, aspects,
            truncation=True, max_length=max_length,
            padding=True, return_tensors="pt",
        )
        input_ids = encodings["input_ids"].to(device)
        attention_mask = encodings["attention_mask"].to(device)

        with torch.no_grad():
            logits = model(input_ids, attention_mask)
            probs = torch.softmax(logits, dim=-1).cpu()

        for j, idx in enumerate(idxs):
            p = probs[j].tolist()
            e = compute_entropy(p)
            if idx not in review_entropies:
                review_entropies[idx] = []
            review_entropies[idx].append(e)

    # average entropy per review across all aspects
    scored = []
    for idx, entropies in review_entropies.items():
        avg_e = sum(entropies) / len(entropies)
        scored.append((reviews[idx]["review_id"], avg_e))

    scored.sort(key=lambda x: -x[1])  # highest entropy first
    return scored


def select_batch(scored_reviews, budget):
    """Select top-K most uncertain reviews."""
    selected_ids = set()
    for review_id, entropy in scored_reviews[:budget]:
        selected_ids.add(review_id)
    return selected_ids


def select_random_batch(reviews, budget, exclude_ids=None):
    """Random baseline for comparison."""
    exclude_ids = exclude_ids or set()
    pool = [r["review_id"] for r in reviews if r["review_id"] not in exclude_ids]
    random.shuffle(pool)
    return set(pool[:budget])


def filter_reviews_by_ids(reviews, ids):
    return [r for r in reviews if r["review_id"] in ids]


def save_selected(reviews, output_path):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        for r in reviews:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def run_selection(model_path, unlabeled_path, output_dir, config_path, budget, strategy="entropy"):
    """One round of active learning: score unlabeled data and select a batch.

    This outputs a JSONL file of selected reviews ready for LLM annotation.
    Does NOT call LLM API — that's done separately with llm_annotator.py.
    """
    with open(config_path) as f:
        config = yaml.safe_load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # load model
    from src.model.classifier import ABSAClassifier
    model = ABSAClassifier(config["training"]["model_name"]).to(device)
    state_dict = torch.load(model_path / "model.pt", map_location=device)
    model.load_state_dict(state_dict)

    tokenizer = AutoTokenizer.from_pretrained(model_path)
    aspect_list = config["aspects"]["general"]

    # load unlabeled data
    reviews = []
    with open(unlabeled_path) as f:
        for line in f:
            if line.strip():
                reviews.append(json.loads(line))
    logger.info("Loaded %d unlabeled reviews", len(reviews))

    # load already-annotated IDs to exclude
    ann_file = PROJECT_ROOT / "data" / "llm_labeled" / "annotations.jsonl"
    already_done = set()
    if ann_file.exists():
        with open(ann_file) as f:
            for line in f:
                if line.strip():
                    already_done.add(json.loads(line)["review_id"])
    logger.info("Excluding %d already-annotated reviews", len(already_done))

    pool = [r for r in reviews if r["review_id"] not in already_done]
    logger.info("Unlabeled pool: %d reviews", len(pool))

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if strategy == "entropy":
        scored = score_unlabeled(model, tokenizer, pool, aspect_list, device)
        selected_ids = select_batch(scored, budget)

        # save scores for analysis
        with open(output_dir / "entropy_scores.json", "w") as f:
            json.dump(scored[:200], f, indent=2)  # top 200 for inspection
    elif strategy == "random":
        selected_ids = select_random_batch(pool, budget)
    else:
        raise ValueError(f"Unknown strategy: {strategy}")

    selected = filter_reviews_by_ids(pool, selected_ids)
    save_selected(selected, output_dir / f"selected_{strategy}_{budget}.jsonl")

    logger.info("Selected %d reviews via %s strategy", len(selected), strategy)

    # print entropy stats
    if strategy == "entropy":
        selected_entropies = [e for rid, e in scored if rid in selected_ids]
        remaining_entropies = [e for rid, e in scored if rid not in selected_ids]
        print(f"\n{'='*50}")
        print(f"  Active Learning Selection (round)")
        print(f"{'='*50}")
        print(f"Strategy:  {strategy}")
        print(f"Budget:    {budget}")
        print(f"Pool size: {len(pool)}")
        print(f"Selected avg entropy:   {sum(selected_entropies)/max(len(selected_entropies),1):.4f}")
        if remaining_entropies:
            print(f"Remaining avg entropy:  {sum(remaining_entropies)/max(len(remaining_entropies),1):.4f}")
        print(f"{'='*50}\n")

    return selected


def run_comparison(model_path, unlabeled_path, config_path, budget):
    """Run both entropy and random selection for comparison."""
    with open(config_path) as f:
        config = yaml.safe_load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    from src.model.classifier import ABSAClassifier
    model = ABSAClassifier(config["training"]["model_name"]).to(device)
    state_dict = torch.load(model_path / "model.pt", map_location=device)
    model.load_state_dict(state_dict)

    tokenizer = AutoTokenizer.from_pretrained(model_path)
    aspect_list = config["aspects"]["general"]

    reviews = []
    with open(unlabeled_path) as f:
        for line in f:
            if line.strip():
                reviews.append(json.loads(line))

    scored = score_unlabeled(model, tokenizer, reviews, aspect_list, device)

    entropy_ids = select_batch(scored, budget)
    random_ids = select_random_batch(reviews, budget)

    overlap = entropy_ids & random_ids

    print(f"\n{'='*50}")
    print(f"  Entropy vs Random Comparison")
    print(f"{'='*50}")
    print(f"Budget: {budget}")
    print(f"Overlap: {len(overlap)} reviews ({len(overlap)/budget*100:.1f}%)")

    entropy_selected = [e for rid, e in scored if rid in entropy_ids]
    random_selected = [e for rid, e in scored if rid in random_ids]
    print(f"Entropy selection avg entropy:  {sum(entropy_selected)/len(entropy_selected):.4f}")
    print(f"Random selection avg entropy:   {sum(random_selected)/len(random_selected):.4f}")
    print(f"{'='*50}\n")


def main():
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")

    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command")

    # select subcommand: run one round
    sel = subparsers.add_parser("select")
    sel.add_argument("--model", type=Path, required=True, help="Path to model checkpoint dir")
    sel.add_argument("--unlabeled", type=Path, required=True, help="Unlabeled reviews JSONL")
    sel.add_argument("--output", type=Path, required=True, help="Output directory")
    sel.add_argument("--budget", type=int, required=True, help="Number of reviews to select")
    sel.add_argument("--strategy", choices=["entropy", "random"], default="entropy")
    sel.add_argument("--config", type=Path, default=PROJECT_ROOT / "configs" / "config.yaml")

    # compare subcommand: entropy vs random
    cmp = subparsers.add_parser("compare")
    cmp.add_argument("--model", type=Path, required=True)
    cmp.add_argument("--unlabeled", type=Path, required=True)
    cmp.add_argument("--budget", type=int, required=True)
    cmp.add_argument("--config", type=Path, default=PROJECT_ROOT / "configs" / "config.yaml")

    args = parser.parse_args()

    if args.command == "select":
        run_selection(args.model, args.unlabeled, args.output, args.config,
                      args.budget, args.strategy)
    elif args.command == "compare":
        run_comparison(args.model, args.unlabeled, args.config, args.budget)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
