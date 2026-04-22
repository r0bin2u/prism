# Prism

Aspect-based sentiment analysis for restaurant reviews. Uses multi-LLM ensemble annotation and distills the knowledge into a DeBERTa model for fast inference.

## Business analogy

Imagine an e-commerce platform that gets millions of product reviews per day. A review like *"Camera is great but battery dies in 3 hours and customer service was rude"* is not just "negative". It contains three different opinions on three different things.

Prism is the engine that turns such unstructured reviews into structured signals. Concrete use cases it maps to:

- **Product page "pros & cons" summary**: auto-generate "users say the camera is great (94%) but complain about battery (67%) and service (58%)" from raw reviews.
- **Recommendation personalization**: a user who cares about *battery* should see products with high battery-positive ratio, regardless of overall rating.
- **Review moderation / risk alerts**: when the *service* aspect suddenly gets 30% more negatives in 24 hours, page the on-call team before customer complaints surface.
- **Customer support ticket routing**: auto-tag incoming complaints by aspect and route to the right team (fulfillment vs product quality vs CS training).

The underlying hard problem is always the same: **"given a piece of unstructured user text, extract fine-grained structured signals in real time at low cost."** Restaurant ABSA is the sandbox I used to validate this pipeline. The same approach transfers to any domain with user-generated text.

## What it does

Given a restaurant review, predicts sentiment for each aspect (food / service / ambience / price / anecdotes).

```
Input:  "The food was amazing but the service was terrible"

Output: food     → positive (0.98)
        service  → negative (0.95)
        ambience → neutral  (0.41)
        price    → neutral  (0.44)
```

## Pipeline

```
SemEval-2014 Restaurant data (3841 human-labeled reviews)
    ↓
3 LLMs annotate each review (Zhipu GLM / SiliconFlow Qwen / Cerebras)
    ↓
Voting aggregation + temperature sharpening → soft label
    ↓
Quality filtering (Cohen's Kappa)
    ↓
DeBERTa-v3-base distillation (mixed CE + KL loss)
    ↓
Temperature calibration (ECE drops from 9.9% to 4.4%)
    ↓
FastAPI service + sliding-window drift monitoring
```

## Results

| Metric | Value |
|---|---|
| Test Macro-F1 | 0.8347 |
| Test Accuracy | 0.8922 |
| ECE reduction | 55% |
| Single inference | 10-20ms |

### Ablation findings

| Variant | Test F1 |
|---|---|
| A: human only | 0.8342 |
| B: + LLM hard label | 0.8084 |
| C: + LLM soft label | 0.8129 |
| D: + sample_weight | 0.8190 |

Interesting result: at 3K samples, human-only actually beats full distillation by 1.5%. But the monotonic B→C→D trend shows each component (soft label, sample weight) is helping. The tipping point is probably around 100K LLM samples.

### Active learning

Did one iteration round after v1: picked 2500 reviews from a 50K Yelp pool using entropy sampling, compared against random 2500.

- Selected samples avg entropy = 0.89, remaining avg = 0.26
- Independent validation: LLM agreement on selected = 0.863 vs random batch = 0.935 (samples the model finds hard, LLMs also find hard)

## Running

### Docker

```bash
docker compose up
```

### Local

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
pip install sentencepiece

python -m src.api.main
```

### Test the API

```bash
curl http://localhost:8000/health

curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"text": "The food was amazing but the service was terrible"}'
```

### Full pipeline

```bash
# prepare data
python -m src.annotation.data_splitter

# LLM annotation (needs API keys)
export ZHIPU_API_KEY="..."
export SILICONFLOW_API_KEY="..."
python -m src.annotation.llm_annotator \
    --input data/splits/train.jsonl \
    --output data/llm_labeled/

# soft labels + filtering
python -m src.annotation.soft_label_builder \
    --llm-labeled data/llm_labeled/ \
    --human-labeled data/splits/train.jsonl \
    --output data/soft_labels/
python -m src.annotation.quality_filter \
    --input data/soft_labels/ \
    --human-ref data/splits/train.jsonl \
    --output data/soft_labels/filtered/

# train
python -m src.model.train

# evaluate
python -m src.evaluation.metrics --checkpoint models/best_model --test-data data/splits/test.jsonl
python -m src.evaluation.calibration --checkpoint models/best_model --val-data data/splits/val.jsonl --test-data data/splits/test.jsonl
python -m src.evaluation.ablation
```

### Tests

```bash
pytest tests/
```

## Layout

```
src/
├── annotation/        # data prep, LLM annotation, soft labels, active learning
├── model/             # DeBERTa classifier, dataset, loss, training
├── evaluation/        # metrics, calibration, ablation
├── api/               # FastAPI service, monitoring
└── features/          # feature engineering, cold-start experiments
tests/                 # pytest tests
configs/config.yaml    # global config
```

## Data sources

- [SemEval-2014 Task 4 Restaurant](https://aclanthology.org/S14-2004.pdf)
- [Yelp Open Dataset](https://business.yelp.com/data/resources/open-dataset/)
