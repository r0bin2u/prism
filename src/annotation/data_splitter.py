"""
Step 1: Download SemEval-2014 Task 4 data, parse XML, convert to JSONL,
and produce stratified train/val/test splits.

Usage:
    python -m src.annotation.data_splitter --config configs/config.yaml
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import xml.etree.ElementTree as ET
from collections import Counter
from pathlib import Path
from urllib.error import URLError
from urllib.request import urlretrieve

import yaml
from sklearn.model_selection import train_test_split

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[2]  # two levels up from src/annotation/

SEMEVAL_URLS: dict[str, dict[str, str]] = {
    "semeval_restaurant": {
        "train": (
            "https://raw.githubusercontent.com/peace195/"
            "aspect-based-sentiment-analysis/master/data/Restaurants_Train.xml"
        ),
        "test": (
            "https://raw.githubusercontent.com/peace195/"
            "aspect-based-sentiment-analysis/master/data/Restaurants_Test_Gold.xml"
        ),
    },
    "semeval_laptop": {
        "train": (
            "https://raw.githubusercontent.com/peace195/"
            "aspect-based-sentiment-analysis/master/data/Laptops_Train.xml"
        ),
        "test": (
            "https://raw.githubusercontent.com/peace195/"
            "aspect-based-sentiment-analysis/master/data/Laptops_Test_Gold.xml"
        ),
    },
}

VALID_SENTIMENTS = {"positive", "negative", "neutral"}

# ---------------------------------------------------------------------------
# Download helpers
# ---------------------------------------------------------------------------


def _download_file(url: str, dest: Path) -> Path:
    """Download *url* to *dest*, skipping if the file already exists."""
    if dest.exists():
        logger.info("Cached: %s", dest)
        return dest

    dest.parent.mkdir(parents=True, exist_ok=True)
    logger.info("Downloading %s -> %s", url, dest)
    try:
        urlretrieve(url, dest)
    except (URLError, OSError) as exc:
        raise RuntimeError(
            f"Failed to download {url}. Check your network connection.\n"
            f"Original error: {exc}"
        ) from exc
    return dest


def download_semeval_data(cache_dir: Path) -> dict[str, dict[str, Path]]:
    """Download all SemEval-2014 XML files, returning paths grouped by dataset
    and split (train / test)."""
    paths: dict[str, dict[str, Path]] = {}
    for dataset, splits in SEMEVAL_URLS.items():
        paths[dataset] = {}
        for split_name, url in splits.items():
            filename = url.rsplit("/", 1)[-1]
            dest = cache_dir / dataset / filename
            _download_file(url, dest)
            paths[dataset][split_name] = dest
    return paths


# ---------------------------------------------------------------------------
# XML parsing
# ---------------------------------------------------------------------------


def parse_semeval_xml(xml_path: Path, dataset: str) -> list[dict]:
    """Parse a SemEval-2014 XML file into a list of review dicts.

    Each dict follows the unified format:
        {
            "review_id": str,
            "text": str,
            "dataset": str,
            "aspects": [{"aspect": str, "sentiment": str, "span": str}, ...]
        }

    Aspect terms with polarity "conflict" are dropped.
    """
    tree = ET.parse(xml_path)
    root = tree.getroot()

    # Derive a short prefix for review IDs, e.g. "semeval_restaurant" -> "rest"
    tag = dataset.split("_")[-1][:4]

    reviews: list[dict] = []
    for idx, sentence in enumerate(root.iter("sentence"), start=1):
        text_el = sentence.find("text")
        if text_el is None or text_el.text is None:
            continue

        text = text_el.text.strip()
        aspects: list[dict] = []

        aspect_terms_el = sentence.find("aspectTerms")
        if aspect_terms_el is not None:
            for at in aspect_terms_el.findall("aspectTerm"):
                polarity = at.get("polarity", "").lower()
                if polarity == "conflict":
                    continue
                if polarity not in VALID_SENTIMENTS:
                    continue
                term = at.get("term", "")
                aspects.append(
                    {
                        "aspect": term.lower(),
                        "sentiment": polarity,
                        "span": term,
                    }
                )

        # Keep the review even if it has no aspect terms (some sentences are
        # aspect-free); downstream code can decide whether to filter them.
        reviews.append(
            {
                "review_id": f"semeval_{tag}_{idx:04d}",
                "text": text,
                "dataset": dataset,
                "aspects": aspects,
            }
        )

    logger.info("Parsed %d reviews from %s", len(reviews), xml_path.name)
    return reviews


# ---------------------------------------------------------------------------
# Stratified splitting
# ---------------------------------------------------------------------------


def _stratification_key(review: dict) -> str:
    """Return the most common aspect-sentiment pair in the review.

    Used as the stratification label for train_test_split.  If the review has
    no aspects, fall back to a generic key so it is still included.
    """
    if not review["aspects"]:
        return "no_aspect"

    counter = Counter(
        (a["aspect"], a["sentiment"]) for a in review["aspects"]
    )
    most_common = counter.most_common(1)[0][0]
    return f"{most_common[0]}|{most_common[1]}"


def stratified_split(
    reviews: list[dict],
    split_ratio: list[float],
    random_state: int = 42,
) -> tuple[list[dict], list[dict], list[dict]]:
    """Split *reviews* into train / val / test using stratification.

    Parameters
    ----------
    split_ratio : list[float]
        Three floats summing to 1.0, e.g. [0.7, 0.15, 0.15].
    """
    train_frac, val_frac, test_frac = split_ratio
    assert abs(train_frac + val_frac + test_frac - 1.0) < 1e-6

    strat_keys = [_stratification_key(r) for r in reviews]

    # Collapse rare keys (fewer than 2 samples) into a catch-all bucket
    # because train_test_split requires at least 2 members per class.
    key_counts = Counter(strat_keys)
    strat_keys = [
        k if key_counts[k] >= 2 else "__rare__" for k in strat_keys
    ]

    # First split: train vs (val + test)
    val_test_frac = val_frac + test_frac
    train_data, temp_data, train_keys, temp_keys = train_test_split(
        reviews,
        strat_keys,
        test_size=val_test_frac,
        random_state=random_state,
        stratify=strat_keys,
    )

    # Second split: val vs test  (proportion of test within the temp set)
    test_relative = test_frac / val_test_frac

    # Re-check for rare keys in the temp set
    temp_key_counts = Counter(temp_keys)
    temp_keys_safe = [
        k if temp_key_counts[k] >= 2 else "__rare__" for k in temp_keys
    ]

    val_data, test_data = train_test_split(
        temp_data,
        test_size=test_relative,
        random_state=random_state,
        stratify=temp_keys_safe,
    )

    return train_data, val_data, test_data


# ---------------------------------------------------------------------------
# I/O helpers
# ---------------------------------------------------------------------------


def save_jsonl(records: list[dict], path: Path) -> None:
    """Write a list of dicts as JSONL."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        for record in records:
            fh.write(json.dumps(record, ensure_ascii=False) + "\n")
    logger.info("Saved %d records to %s", len(records), path)


def sha256_of_file(path: Path) -> str:
    """Return the hex SHA-256 digest of *path*."""
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


# ---------------------------------------------------------------------------
# Summary statistics
# ---------------------------------------------------------------------------


def print_summary(
    all_reviews: list[dict],
    train: list[dict],
    val: list[dict],
    test: list[dict],
) -> None:
    """Print human-readable summary statistics."""
    print("\n" + "=" * 60)
    print("  Data Splitting Summary")
    print("=" * 60)

    print(f"\nTotal reviews: {len(all_reviews)}")

    # Per-dataset counts
    ds_counts = Counter(r["dataset"] for r in all_reviews)
    for ds, cnt in sorted(ds_counts.items()):
        print(f"  {ds}: {cnt}")

    # Split sizes
    print(f"\nSplit sizes:")
    print(f"  train : {len(train)}")
    print(f"  val   : {len(val)}")
    print(f"  test  : {len(test)}")

    # Aspect-sentiment distribution (across all reviews)
    aspect_sent = Counter()
    for r in all_reviews:
        for a in r["aspects"]:
            aspect_sent[(a["sentiment"],)] += 1

    print(f"\nAspect-sentiment distribution (all data):")
    for (sent,), cnt in sorted(aspect_sent.items(), key=lambda x: -x[1]):
        print(f"  {sent}: {cnt}")

    print("=" * 60 + "\n")


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------


def load_config(config_path: Path) -> dict:
    with open(config_path, "r", encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def run(config_path: Path) -> None:
    config = load_config(config_path)

    cache_dir = PROJECT_ROOT / "data" / "human_labeled"
    splits_dir = PROJECT_ROOT / Path(config["data"]["splits_dir"])
    split_ratio: list[float] = config["data"]["split_ratio"]
    valid_sentiments = set(config.get("sentiments", VALID_SENTIMENTS))

    # 1. Download
    print("Downloading SemEval-2014 data ...")
    xml_paths = download_semeval_data(cache_dir)

    # 2. Parse
    all_reviews: list[dict] = []
    for dataset, split_paths in xml_paths.items():
        for _split_name, xml_path in split_paths.items():
            reviews = parse_semeval_xml(xml_path, dataset)
            all_reviews.extend(reviews)

    # Filter aspects to only valid sentiments (should already be handled,
    # but this is a safety net).
    for review in all_reviews:
        review["aspects"] = [
            a for a in review["aspects"] if a["sentiment"] in valid_sentiments
        ]

    print(f"Parsed {len(all_reviews)} reviews total.")

    # 3. Stratified split
    train, val, test = stratified_split(all_reviews, split_ratio)

    # 4. Save splits
    save_jsonl(train, splits_dir / "train.jsonl")
    save_jsonl(val, splits_dir / "val.jsonl")
    save_jsonl(test, splits_dir / "test.jsonl")

    # 5. SHA-256 of test set for reproducibility
    test_hash = sha256_of_file(splits_dir / "test.jsonl")
    print(f"\ntest.jsonl SHA-256: {test_hash}")

    # 6. Summary
    print_summary(all_reviews, train, val, test)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    parser = argparse.ArgumentParser(
        description="Download, parse, and split SemEval-2014 Task 4 data.",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=PROJECT_ROOT / "configs" / "config.yaml",
        help="Path to the YAML config file (default: configs/config.yaml).",
    )
    args = parser.parse_args()

    if not args.config.exists():
        parser.error(f"Config file not found: {args.config}")

    run(args.config)


if __name__ == "__main__":
    main()
