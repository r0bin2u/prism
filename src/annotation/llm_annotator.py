"""
LLM offline annotation for reviews.
Each review gets N annotation runs with per-run provider/model config.

Supported providers:
- gemini      (Google Gemini, native SDK, GEMINI_API_KEYS comma-separated)
- groq        (Groq, OpenAI-compat, GROQ_API_KEY)
- zhipu       (Zhipu GLM, OpenAI-compat, ZHIPU_API_KEY)
- siliconflow (SiliconFlow, OpenAI-compat, SILICONFLOW_API_KEY)
- cerebras    (Cerebras Cloud, OpenAI-compat, CEREBRAS_API_KEY)

Usage:
    python -m src.annotation.llm_annotator \
        --input data/splits/train.jsonl \
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

class GeminiKeyRotator:
    def __init__(self):
        import os
        raw = os.environ.get("GEMINI_API_KEYS", os.environ.get("GEMINI_API_KEY", ""))
        self.keys = [k.strip() for k in raw.split(",") if k.strip()]
        if not self.keys:
            raise ValueError("Set GEMINI_API_KEYS (comma-separated) or GEMINI_API_KEY env var")
        self._idx = 0
        self._clients = {}

    def next_client(self):
        key = self.keys[self._idx % len(self.keys)]
        self._idx += 1
        if key not in self._clients:
            from google import genai
            self._clients[key] = genai.Client(api_key=key)
        return self._clients[key]


def call_gemini(rotator, model, system, user, temperature):
    client = rotator.next_client()
    resp = client.models.generate_content(
        model=model,
        contents=user,
        config={
            "system_instruction": system,
            "temperature": temperature,
            "max_output_tokens": 512,
        },
    )
    return resp.text


def call_openai_compat(client, model, system, user, temperature):
    resp = client["http"].post(
        f"{client['base_url']}/chat/completions",
        headers={
            "Authorization": f"Bearer {client['api_key']}",
            "Content-Type": "application/json",
        },
        json={
            "model": model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "temperature": temperature,
            "max_tokens": 512,
        },
    )
    resp.raise_for_status()
    data = resp.json()
    return data["choices"][0]["message"]["content"]


OPENAI_COMPAT_PROVIDERS = {
    "groq":        ("https://api.groq.com/openai/v1",      "GROQ_API_KEY"),
    "zhipu":       ("https://open.bigmodel.cn/api/paas/v4", "ZHIPU_API_KEY"),
    "siliconflow": ("https://api.siliconflow.cn/v1",       "SILICONFLOW_API_KEY"),
    "cerebras":    ("https://api.cerebras.ai/v1",          "CEREBRAS_API_KEY"),
}


def make_client(provider):
    if provider == "gemini":
        return GeminiKeyRotator(), call_gemini
    if provider not in OPENAI_COMPAT_PROVIDERS:
        raise ValueError(f"Unknown provider: {provider}")

    import os
    import httpx
    base_url, env_var = OPENAI_COMPAT_PROVIDERS[provider]
    api_key = os.environ.get(env_var)
    if not api_key:
        raise ValueError(f"Set {env_var} env var for provider '{provider}'")
    client = {
        "http": httpx.Client(timeout=60.0),
        "base_url": base_url,
        "api_key": api_key,
    }
    return client, call_openai_compat


# -- JSON parsing --

def parse_llm_response(raw: str, valid_aspects: set, valid_sentiments: set):
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
            wait = 2 ** attempt
            logger.warning("Retry %d/%d after %ds: %s", attempt + 1, max_retries, wait, e)
            time.sleep(wait)


# -- Core annotation --

def annotate_one_review(review_text, aspect_list, run_configs, temperature, valid_aspects, valid_sentiments):
    system = SYSTEM_PROMPT.format(aspect_list=", ".join(aspect_list))
    user = USER_PROMPT.format(review_text=review_text)

    annotations = []
    for run_id, rc in enumerate(run_configs):
        raw = call_with_retry(rc["call_fn"], rc["client"], rc["model_name"], system, user, temperature)
        if raw is None:
            annotations.append({
                "run_id": run_id,
                "provider": rc["provider"],
                "model": rc["model_name"],
                "raw_response": "",
                "parsed": [],
                "parse_success": False,
            })
        else:
            parsed, success = parse_llm_response(raw, valid_aspects, valid_sentiments)
            annotations.append({
                "run_id": run_id,
                "provider": rc["provider"],
                "model": rc["model_name"],
                "raw_response": raw,
                "parsed": parsed,
                "parse_success": success,
            })

        if run_id < len(run_configs) - 1:
            time.sleep(rc.get("sleep", 4))

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

def run(input_path: Path, output_dir: Path, config_path: Path, max_reviews: int = 0):
    with open(config_path) as f:
        config = yaml.safe_load(f)

    llm_cfg = config["llm"]
    temperature = llm_cfg["temperature"]
    save_interval = llm_cfg["batch_save_interval"]

    # build per-run configs
    run_configs = []
    for r in llm_cfg["runs"]:
        client, call_fn = make_client(r["provider"])
        run_configs.append({
            "provider": r["provider"],
            "model_name": r["model_name"],
            "client": client,
            "call_fn": call_fn,
            "sleep": r.get("sleep_seconds", 4),
        })

    aspect_list = config["aspects"]["restaurant"]
    valid_aspects = set(a.lower() for a in aspect_list)
    valid_sentiments = set(config["sentiments"])

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
    new_done = 0
    total_parse_success = 0
    total_parse_attempts = 0
    aspect_counter = {}

    for i, review in enumerate(reviews):
        rid = review["review_id"]
        if rid in processed_ids:
            continue

        text = review.get("text", "")
        annotations = annotate_one_review(
            text, aspect_list, run_configs, temperature, valid_aspects, valid_sentiments,
        )

        record = {
            "review_id": rid,
            "text": text,
            "llm_annotations": annotations,
        }

        with open(output_file, "a") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

        total_done += 1
        new_done += 1
        for ann in annotations:
            total_parse_attempts += 1
            if ann["parse_success"]:
                total_parse_success += 1
            for item in ann["parsed"]:
                asp = item["aspect"]
                aspect_counter[asp] = aspect_counter.get(asp, 0) + 1

        if total_done % save_interval == 0:
            rate = total_parse_success / max(total_parse_attempts, 1) * 100
            logger.info(
                "Progress: %d/%d done | parse success rate: %.1f%%",
                total_done, len(reviews), rate,
            )

        if max_reviews and new_done >= max_reviews:
            logger.info("Reached --max-reviews %d, stopping.", max_reviews)
            break

        # rate limit between reviews
        time.sleep(2)

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
    parser.add_argument("--max-reviews", type=int, default=0, help="Max reviews to process (0 = all)")
    args = parser.parse_args()

    if not args.input.exists():
        parser.error(f"Input file not found: {args.input}")

    run(args.input, args.output, args.config, max_reviews=args.max_reviews)


if __name__ == "__main__":
    main()
