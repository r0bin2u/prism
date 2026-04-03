from __future__ import annotations

import argparse
import json
import logging
from collections import Counter, defaultdict
from pathlib import Path

import yaml

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[2]

SENTIMENT_ORDER = ["positive", "negative", "neutral"]
SENTIMENT_TO_IDX = {s: i for i, s in enumerate(SENTIMENT_ORDER)}


def sharpen(dist, T):
    powered = [v ** (1.0 / T) for v in dist]
    total = sum(powered)
    if total == 0:
        return dist
    return [v / total for v in powered]


def build_soft_labels_from_llm(annotations, config):
    do_sharpen = config["soft_label"]["temperature_sharpening"]
    T = config["soft_label"]["sharpening_T"]
    num_runs = config["llm"]["num_runs"]

    # for diagnostics
    aspect_mention_counts = defaultdict(list)
    aspect_sentiment_agreements = defaultdict(int)
    aspect_sentiment_totals = defaultdict(int)

    results = []

    for record in annotations:
        successful_runs = [r for r in record["llm_annotations"] if r["parse_success"]]
        n_success = len(successful_runs)
        if n_success == 0:
            continue

        # aspect -> list of sentiments from each run that mentioned it
        aspect_votes = defaultdict(list)
        for run in successful_runs:
            seen = set()
            for item in run["parsed"]:
                asp = item["aspect"]
                if asp not in seen:
                    aspect_votes[asp].append(item["sentiment"])
                    seen.add(asp)

        soft_labels = []
        for aspect, sentiments in aspect_votes.items():
            mention_count = len(sentiments)
            sample_weight = mention_count / n_success

            vote = Counter(sentiments)
            raw_dist = [vote.get(s, 0) / mention_count for s in SENTIMENT_ORDER]

            label = sharpen(raw_dist, T) if do_sharpen else raw_dist
            assert abs(sum(label) - 1.0) < 1e-6

            majority = SENTIMENT_ORDER[label.index(max(label))]

            soft_labels.append({
                "aspect": aspect,
                "label": [round(v, 6) for v in label],
                "sample_weight": round(sample_weight, 4),
                "num_mentions": mention_count,
                "majority_vote": majority,
            })

            # track diagnostics
            aspect_mention_counts[aspect].append(mention_count)
            aspect_sentiment_totals[aspect] += 1
            if len(set(sentiments)) == 1:
                aspect_sentiment_agreements[aspect] += 1

        if soft_labels:
            results.append({
                "review_id": record["review_id"],
                "text": record["text"],
                "soft_labels": soft_labels,
                "source": "llm",
            })

    # diagnostics report
    per_aspect = {}
    for aspect in aspect_sentiment_totals:
        t = aspect_sentiment_totals[aspect]
        a = aspect_sentiment_agreements[aspect]
        mentions = aspect_mention_counts[aspect]
        avg_mentions = sum(mentions) / len(mentions) if mentions else 0
        per_aspect[aspect] = {
            "sentiment_agreement_rate": round(a / t, 3) if t > 0 else 0,
            "avg_mention_rate": round(avg_mentions / num_runs, 3),
            "annotation_count": t,
        }

    overall_agree = sum(aspect_sentiment_agreements.values())
    overall_total = sum(aspect_sentiment_totals.values())
    diagnostics = {
        "total_reviews": len(results),
        "total_aspect_annotations": sum(len(r["soft_labels"]) for r in results),
        "overall_sentiment_agreement": round(overall_agree / max(overall_total, 1), 3),
        "per_aspect_reliability": per_aspect,
    }

    return results, diagnostics


def build_soft_labels_from_human(human_path):
    results = []
    with open(human_path) as f:
        for line in f:
            record = json.loads(line.strip())
            soft_labels = []
            for asp in record.get("aspects", []):
                sent = asp["sentiment"]
                if sent not in SENTIMENT_TO_IDX:
                    continue
                one_hot = [0.0, 0.0, 0.0]
                one_hot[SENTIMENT_TO_IDX[sent]] = 1.0
                soft_labels.append({
                    "aspect": asp["aspect"],
                    "label": one_hot,
                    "sample_weight": 1.0,
                    "num_mentions": 1,
                    "majority_vote": sent,
                })
            if soft_labels:
                results.append({
                    "review_id": record["review_id"],
                    "text": record["text"],
                    "soft_labels": soft_labels,
                    "source": "human",
                })
    return results


def save_jsonl(records, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def run(llm_dir, human_path, output_dir, config_path):
    with open(config_path) as f:
        config = yaml.safe_load(f)

    # load LLM annotations
    ann_file = llm_dir / "annotations.jsonl"
    annotations = []
    if ann_file.exists():
        with open(ann_file) as f:
            for line in f:
                if line.strip():
                    annotations.append(json.loads(line))
    logger.info("Loaded %d LLM annotations", len(annotations))

    llm_results, diagnostics = build_soft_labels_from_llm(annotations, config)
    logger.info("Built %d LLM soft label records", len(llm_results))

    human_results = []
    if human_path.exists():
        human_results = build_soft_labels_from_human(human_path)
        logger.info("Built %d human soft label records", len(human_results))

    # save everything
    output_dir.mkdir(parents=True, exist_ok=True)
    if llm_results:
        save_jsonl(llm_results, output_dir / "llm_soft_labels.jsonl")
    if human_results:
        save_jsonl(human_results, output_dir / "human_soft_labels.jsonl")

    with open(output_dir / "annotation_diagnostics.json", "w") as f:
        json.dump(diagnostics, f, indent=2)

    # print summary
    print(f"\n{'='*55}")
    print(f"  Soft Label Builder Summary")
    print(f"{'='*55}")
    print(f"LLM soft labels:   {len(llm_results)} reviews")
    print(f"Human soft labels:  {len(human_results)} reviews")
    print(f"Overall sentiment agreement: {diagnostics['overall_sentiment_agreement']}")
    print(f"\nPer-aspect reliability:")
    for asp, info in sorted(diagnostics["per_aspect_reliability"].items(),
                            key=lambda x: -x[1]["sentiment_agreement_rate"]):
        print(f"  {asp:30s} agree={info['sentiment_agreement_rate']:.3f}  "
              f"mention={info['avg_mention_rate']:.3f}  n={info['annotation_count']}")
    print(f"{'='*55}\n")


def main():
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
    parser = argparse.ArgumentParser()
    parser.add_argument("--llm-labeled", type=Path, required=True)
    parser.add_argument("--human-labeled", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=PROJECT_ROOT / "configs" / "config.yaml")
    args = parser.parse_args()
    run(args.llm_labeled, args.human_labeled, args.output, args.config)


if __name__ == "__main__":
    main()
