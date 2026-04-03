"""
LLM offline annotation for Amazon reviews.
Each review gets N annotation runs (default 3) at temperature=0.7.
Supports Anthropic Claude and OpenAI GPT, switchable via config.

Usage:
    python -m src.annotation.llm_annotator \
        --input data/raw/amazon_reviews.jsonl \
        --output data/llm_labeled/ \
        --config configs/config.yaml
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import time
from pathlib import Path

import yaml

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[2]

SYSTEM_PROMPT = """\
You are a professional product review annotator.
Your task is to identify aspects mentioned in the review and classify the sentiment for each aspect.

Rules:
- Only use aspects from this list: {aspect_list}
- Sentiment must be one of: positive, negative, neutral
- If an aspect is not explicitly mentioned, do NOT annotate it
- One review may have multiple aspects or zero aspects
- Output ONLY valid JSON, no other text

Examples:
Review: "The battery life is amazing but the screen is too dim"
Output: [{{"aspect": "battery", "sentiment": "positive"}}, {{"aspect": "screen", "sentiment": "negative"}}]

Review: "Fast delivery, everything arrived in good condition"
Output: [{{"aspect": "shipping", "sentiment": "positive"}}, {{"aspect": "packaging", "sentiment": "positive"}}]"""

USER_PROMPT = """\
Review: "{review_text}"

Output format (JSON array):
[
  {{"aspect": "...", "sentiment": "positive/negative/neutral"}}
]

If no aspect is mentioned, output: []"""


# -- LLM API calls --

def call_anthropic(client, model, system, user, temperature):
    resp = client.messages.create(
        model=model,
        max_tokens=512,
        temperature=temperature,
        system=system,
        messages=[{"role": "user", "content": user}],
    )
    return resp.content[0].text


def call_openai(client, model, system, user, temperature):
    resp = client.chat.completions.create(
        model=model,
        max_tokens=512,
        temperature=temperature,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
    )
    return resp.choices[0].message.content


def make_client(provider):
    if provider == "anthropic":
        import anthropic
        return anthropic.Anthropic(), call_anthropic
    elif provider == "openai":
        import openai
        return openai.OpenAI(), call_openai
    else:
        raise ValueError(f"Unknown provider: {provider}")


# -- JSON parsing --

def parse_llm_response(raw: str, valid_aspects: set, valid_sentiments: set):
    """Try to extract JSON array from LLM response. Returns (parsed_list, success)."""
    text = raw.strip()

    # strip markdown code fences
    m = re.search(r"```(?:json)?\s*(.*?)\s*```", text, re.DOTALL)
    if m:
        text = m.group(1)

    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        return [], False

    if not isinstance(data, list):
        return [], False

    results = []
    for item in data:
        if not isinstance(item, dict):
            continue
        aspect = item.get("aspect", "").lower().strip()
        sentiment = item.get("sentiment", "").lower().strip()

        if aspect not in valid_aspects:
            logger.debug("Filtered unknown aspect: %s", aspect)
            continue
        if sentiment not in valid_sentiments:
            logger.debug("Filtered unknown sentiment: %s", sentiment)
            continue
        results.append({"aspect": aspect, "sentiment": sentiment})

    return results, True


# -- Retry logic --

def call_with_retry(fn, client, model, system, user, temperature, max_retries=3):
    for attempt in range(max_retries + 1):
        try:
            return fn(client, model, system, user, temperature)
        except Exception as e:
            if attempt == max_retries:
                logger.error("Failed after %d retries: %s", max_retries, e)
                return None
            wait = 2 ** attempt  # 1s, 2s, 4s
            logger.warning("Retry %d/%d after %ds: %s", attempt + 1, max_retries, wait, e)
            time.sleep(wait)


# -- Core annotation --

def annotate_one_review(review_text, aspect_list, client, call_fn, model, temperature, num_runs, valid_aspects, valid_sentiments):
    system = SYSTEM_PROMPT.format(aspect_list=", ".join(aspect_list))
    user = USER_PROMPT.format(review_text=review_text)

    annotations = []
    for run_id in range(num_runs):
        raw = call_with_retry(call_fn, client, model, system, user, temperature)
        if raw is None:
            annotations.append({
                "run_id": run_id,
                "raw_response": "",
                "parsed": [],
                "parse_success": False,
            })
        else:
            parsed, success = parse_llm_response(raw, valid_aspects, valid_sentiments)
            annotations.append({
                "run_id": run_id,
                "raw_response": raw,
                "parsed": parsed,
                "parse_success": success,
            })

        if run_id < num_runs - 1:
            time.sleep(0.5)

    return annotations


# -- Checkpoint: load already processed IDs --

def load_processed_ids(output_file: Path) -> set:
    ids = set()
    if output_file.exists():
        with open(output_file) as f:
            for line in f:
                line = line.strip()
                if line:
                    record = json.loads(line)
                    ids.add(record["review_id"])
    return ids


# -- Main --

def run(input_path: Path, output_dir: Path, config_path: Path):
    with open(config_path) as f:
        config = yaml.safe_load(f)

    llm_cfg = config["llm"]
    provider = llm_cfg["provider"]
    model = llm_cfg["model_name"]
    temperature = llm_cfg["temperature"]
    num_runs = llm_cfg["num_runs"]
    save_interval = llm_cfg["batch_save_interval"]

    aspect_list = config["aspects"]["general"]
    valid_aspects = set(a.lower() for a in aspect_list)
    valid_sentiments = set(config["sentiments"])

    client, call_fn = make_client(provider)

    # load input reviews
    reviews = []
    with open(input_path) as f:
        for line in f:
            line = line.strip()
            if line:
                reviews.append(json.loads(line))

    logger.info("Loaded %d reviews from %s", len(reviews), input_path)

    # checkpoint
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / "annotations.jsonl"
    processed_ids = load_processed_ids(output_file)
    if processed_ids:
        logger.info("Resuming: skipping %d already processed reviews", len(processed_ids))

    # stats
    total_done = len(processed_ids)
    total_parse_success = 0
    total_parse_attempts = 0
    aspect_counter = {}

    for i, review in enumerate(reviews):
        rid = review["review_id"]
        if rid in processed_ids:
            continue

        text = review.get("text", "")
        annotations = annotate_one_review(
            text, aspect_list, client, call_fn, model,
            temperature, num_runs, valid_aspects, valid_sentiments,
        )

        record = {
            "review_id": rid,
            "text": text,
            "llm_annotations": annotations,
        }

        # append to file immediately
        with open(output_file, "a") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

        # update stats
        total_done += 1
        for ann in annotations:
            total_parse_attempts += 1
            if ann["parse_success"]:
                total_parse_success += 1
            for item in ann["parsed"]:
                asp = item["aspect"]
                aspect_counter[asp] = aspect_counter.get(asp, 0) + 1

        # progress log
        if total_done % save_interval == 0:
            rate = total_parse_success / max(total_parse_attempts, 1) * 100
            logger.info(
                "Progress: %d/%d done | parse success rate: %.1f%%",
                total_done, len(reviews), rate,
            )

        time.sleep(0.5)  # rate limiting between reviews

    # final summary
    rate = total_parse_success / max(total_parse_attempts, 1) * 100
    print(f"\n{'='*50}")
    print(f"  Annotation Summary")
    print(f"{'='*50}")
    print(f"Total reviews:     {total_done}")
    print(f"Parse success rate: {rate:.1f}%")
    print(f"\nAspect frequency:")
    for asp, cnt in sorted(aspect_counter.items(), key=lambda x: -x[1]):
        print(f"  {asp}: {cnt}")
    print(f"{'='*50}\n")


def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    parser = argparse.ArgumentParser(description="LLM offline annotation for reviews")
    parser.add_argument("--input", type=Path, required=True, help="Input JSONL file")
    parser.add_argument("--output", type=Path, required=True, help="Output directory")
    parser.add_argument("--config", type=Path, default=PROJECT_ROOT / "configs" / "config.yaml")
    args = parser.parse_args()

    if not args.input.exists():
        parser.error(f"Input file not found: {args.input}")

    run(args.input, args.output, args.config)


if __name__ == "__main__":
    main()
