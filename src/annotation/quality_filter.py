from __future__ import annotations

import argparse
import json
import logging
from collections import defaultdict
from pathlib import Path

import yaml

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[2]

SENTIMENT_ORDER = ["positive", "negative", "neutral"]


def load_jsonl(path):
    records = []
    with open(path) as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line))
    return records


def save_jsonl(records, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def load_llm_parse_stats(llm_dir):
    """Returns {review_id: number of parse failures} from raw LLM annotations."""
    stats = {}
    ann_file = llm_dir / "annotations.jsonl"
    if not ann_file.exists():
        return stats
    with open(ann_file) as f:
        for line in f:
            if not line.strip():
                continue
            record = json.loads(line)
            fails = sum(1 for r in record["llm_annotations"] if not r["parse_success"])
            stats[record["review_id"]] = fails
    return stats


# -- Filtering --

def filter_consistency(record):
    """Remove aspects where all 3 sentiments got equal votes (1:1:1).
    Returns filtered record or None if all aspects were dropped."""
    kept = []
    dropped = 0
    for sl in record["soft_labels"]:
        label = sl["label"]
        # 1:1:1 means all three values are equal (each ~0.333)
        nonzero = [v for v in label if v > 0]
        if len(nonzero) == 3 and max(label) - min(label) < 0.05:
            dropped += 1
            logger.debug("Dropped %s aspect=%s (1:1:1 vote)", record["review_id"], sl["aspect"])
        else:
            kept.append(sl)

    if not kept:
        return None, dropped
    record = dict(record)
    record["soft_labels"] = kept
    return record, dropped


def filter_records(llm_records, parse_stats):
    """Apply all filters, return (filtered_records, drop_summary)."""
    drop_reasons = {"consistency": 0, "parse_failure": 0, "all_aspects_dropped": 0}
    filtered = []

    for record in llm_records:
        rid = record["review_id"]

        # parse failure filter: >1 failed parse out of N runs
        if parse_stats.get(rid, 0) > 1:
            drop_reasons["parse_failure"] += 1
            logger.debug("Dropped %s: too many parse failures", rid)
            continue

        # consistency filter
        record_filtered, n_dropped = filter_consistency(record)
        drop_reasons["consistency"] += n_dropped

        if record_filtered is None:
            drop_reasons["all_aspects_dropped"] += 1
            logger.debug("Dropped %s: all aspects removed by consistency filter", rid)
            continue

        filtered.append(record_filtered)

    return filtered, drop_reasons


# -- Per-aspect calibration against human data --

def compute_per_aspect_kappa(llm_records, human_records):
    """Compute Cohen's Kappa per aspect using overlapping reviews."""
    # build lookup: review_id -> {aspect: majority_vote} for both sources
    llm_lookup = {}
    for r in llm_records:
        llm_lookup[r["review_id"]] = {
            sl["aspect"]: sl["majority_vote"] for sl in r["soft_labels"]
        }

    human_lookup = {}
    for r in human_records:
        human_lookup[r["review_id"]] = {
            sl["aspect"]: sl["majority_vote"] for sl in r["soft_labels"]
        }

    # find overlapping review IDs
    overlap_ids = set(llm_lookup.keys()) & set(human_lookup.keys())
    if not overlap_ids:
        logger.info("No overlapping reviews between LLM and human for calibration")
        return {}

    # collect paired labels per aspect
    aspect_pairs = defaultdict(list)  # aspect -> [(human_label, llm_label), ...]
    for rid in overlap_ids:
        human_aspects = human_lookup[rid]
        llm_aspects = llm_lookup[rid]
        for aspect in set(human_aspects.keys()) & set(llm_aspects.keys()):
            aspect_pairs[aspect].append((human_aspects[aspect], llm_aspects[aspect]))

    # compute kappa per aspect
    report = {}
    for aspect, pairs in aspect_pairs.items():
        if len(pairs) < 5:
            report[aspect] = {"kappa": None, "n": len(pairs), "note": "too few samples"}
            continue

        kappa = cohens_kappa(pairs)
        report[aspect] = {
            "kappa": round(kappa, 3),
            "n": len(pairs),
            "accuracy": round(sum(1 for h, l in pairs if h == l) / len(pairs), 3),
        }

        if kappa < 0.6:
            logger.warning("Low Kappa for aspect '%s': %.3f (n=%d)", aspect, kappa, len(pairs))

    return report


def cohens_kappa(pairs):
    """Compute Cohen's Kappa from a list of (label_a, label_b) pairs."""
    n = len(pairs)
    if n == 0:
        return 0.0

    # count agreements
    agree = sum(1 for a, b in pairs if a == b)
    po = agree / n  # observed agreement

    # count label frequencies for each rater
    a_counts = defaultdict(int)
    b_counts = defaultdict(int)
    for a, b in pairs:
        a_counts[a] += 1
        b_counts[b] += 1

    # expected agreement by chance
    all_labels = set(a_counts.keys()) | set(b_counts.keys())
    pe = sum((a_counts[l] / n) * (b_counts[l] / n) for l in all_labels)

    if pe >= 1.0:
        return 1.0
    return (po - pe) / (1 - pe)


def run(input_dir, human_ref_path, llm_dir, output_dir, config_path):
    with open(config_path) as f:
        config = yaml.safe_load(f)

    # load soft labels
    llm_file = input_dir / "llm_soft_labels.jsonl"
    llm_records = load_jsonl(llm_file) if llm_file.exists() else []

    human_file = input_dir / "human_soft_labels.jsonl"
    human_records = load_jsonl(human_file) if human_file.exists() else []

    # load parse stats from raw LLM annotations
    parse_stats = load_llm_parse_stats(llm_dir)

    # filter LLM records
    original_count = len(llm_records)
    filtered_llm, drop_reasons = filter_records(llm_records, parse_stats)

    # per-aspect calibration
    kappa_report = compute_per_aspect_kappa(filtered_llm, human_records)

    # save filtered data (human passes through unfiltered)
    output_dir.mkdir(parents=True, exist_ok=True)
    if filtered_llm:
        save_jsonl(filtered_llm, output_dir / "llm_soft_labels.jsonl")
    if human_records:
        save_jsonl(human_records, output_dir / "human_soft_labels.jsonl")

    # save calibration report
    if kappa_report:
        with open(output_dir / "calibration_report.json", "w") as f:
            json.dump(kappa_report, f, indent=2)

    # print summary
    dropped_total = original_count - len(filtered_llm)
    drop_pct = dropped_total / max(original_count, 1) * 100

    print(f"\n{'='*55}")
    print(f"  Quality Filter Summary")
    print(f"{'='*55}")
    print(f"LLM records:  {original_count} -> {len(filtered_llm)} ({drop_pct:.1f}% dropped)")
    print(f"Human records: {len(human_records)} (passed through)")
    print(f"\nDrop reasons:")
    for reason, cnt in drop_reasons.items():
        print(f"  {reason}: {cnt}")

    if kappa_report:
        print(f"\nPer-aspect Kappa (LLM vs Human):")
        for asp, info in sorted(kappa_report.items()):
            k = info.get("kappa")
            if k is not None:
                flag = " ⚠ LOW" if k < 0.6 else ""
                print(f"  {asp:30s} kappa={k:.3f}  acc={info['accuracy']:.3f}  n={info['n']}{flag}")
            else:
                print(f"  {asp:30s} {info.get('note', 'N/A')}")

    print(f"{'='*55}\n")


def main():
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True, help="soft_labels dir")
    parser.add_argument("--human-ref", type=Path, required=True)
    parser.add_argument("--llm-dir", type=Path, default=PROJECT_ROOT / "data" / "llm_labeled",
                        help="raw LLM annotations dir (for parse_success stats)")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=PROJECT_ROOT / "configs" / "config.yaml")
    args = parser.parse_args()
    run(args.input, args.human_ref, args.llm_dir, args.output, args.config)


if __name__ == "__main__":
    main()
