from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch
import yaml
from fastapi import HTTPException, Request
from transformers import AutoTokenizer

from src.api.monitoring import PredictionMonitor
from src.api.schemas import AspectPrediction
from src.model.classifier import ABSAClassifier

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[2]

SENTIMENTS = ["positive", "negative", "neutral"]
DEFAULT_ASPECTS = ["food", "service", "ambience", "price", "anecdotes/miscellaneous"]


@dataclass
class ModelBundle:
    classifier: ABSAClassifier
    tokenizer: object
    device: torch.device
    calibration_T: float
    calibration_loaded: bool
    model_version: str
    max_length: int
    mini_batch_size: int
    default_aspects: list[str] = field(default_factory=lambda: list(DEFAULT_ASPECTS))

    @torch.no_grad()
    def _forward(self, texts: list[str], aspects: list[str]) -> np.ndarray:
        enc = self.tokenizer(
            texts, aspects,
            truncation=True, max_length=self.max_length,
            padding=True, return_tensors="pt",
        )
        input_ids = enc["input_ids"].to(self.device)
        attention_mask = enc["attention_mask"].to(self.device)
        logits = self.classifier(input_ids, attention_mask)
        if self.calibration_loaded:
            logits = logits / self.calibration_T
        probs = torch.softmax(logits, dim=-1)
        return probs.cpu().numpy()

    def predict_aspects(self, text: str, aspects: list[str] | None = None) -> list[AspectPrediction]:
        asps = list(aspects) if aspects else list(self.default_aspects)
        probs = self._forward([text] * len(asps), asps)
        return [_row_to_pred(asps[i], probs[i]) for i in range(len(asps))]

    def predict_batch(self, reviews: list[tuple[str, list[str] | None]]) -> list[list[AspectPrediction]]:
        flat_texts: list[str] = []
        flat_aspects: list[str] = []
        owners: list[int] = []
        for idx, (text, asps) in enumerate(reviews):
            asps = list(asps) if asps else list(self.default_aspects)
            for a in asps:
                flat_texts.append(text)
                flat_aspects.append(a)
                owners.append(idx)

        if not flat_texts:
            return [[] for _ in reviews]

        chunks = []
        bs = self.mini_batch_size
        for start in range(0, len(flat_texts), bs):
            chunks.append(self._forward(flat_texts[start:start + bs], flat_aspects[start:start + bs]))
        probs = np.concatenate(chunks, axis=0)

        results: list[list[AspectPrediction]] = [[] for _ in reviews]
        for i, owner in enumerate(owners):
            results[owner].append(_row_to_pred(flat_aspects[i], probs[i]))
        return results


def _row_to_pred(aspect: str, prob_row: np.ndarray) -> AspectPrediction:
    idx = int(prob_row.argmax())
    return AspectPrediction(
        aspect=aspect,
        sentiment=SENTIMENTS[idx],
        confidence=float(prob_row[idx]),
        probabilities={SENTIMENTS[j]: float(prob_row[j]) for j in range(3)},
    )


def load_model_bundle(checkpoint_dir: Path, config: dict) -> ModelBundle:
    checkpoint_dir = Path(checkpoint_dir)
    if not checkpoint_dir.exists():
        raise FileNotFoundError(f"checkpoint dir not found: {checkpoint_dir}")

    weights_path = checkpoint_dir / "model.pt"
    if not weights_path.exists():
        raise FileNotFoundError(f"model.pt not found in {checkpoint_dir}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("loading model on %s", device)

    classifier = ABSAClassifier(config["training"]["model_name"]).to(device)
    state_dict = torch.load(weights_path, map_location=device)
    classifier.load_state_dict(state_dict)
    classifier.eval()

    tokenizer = AutoTokenizer.from_pretrained(checkpoint_dir)

    cal_path = checkpoint_dir / "calibration_T.json"
    if cal_path.exists():
        with open(cal_path) as f:
            cal_data = json.load(f)
        T = float(cal_data["temperature"])
        cal_loaded = True
        logger.info("calibration loaded, T=%.4f", T)
    else:
        T = 1.0
        cal_loaded = False
        logger.warning("calibration_T.json not found in %s, using T=1.0", checkpoint_dir)

    versions_path = PROJECT_ROOT / "configs" / "versions.yaml"
    model_version = "unknown"
    try:
        with open(versions_path) as f:
            versions = yaml.safe_load(f)
        model_version = versions.get("api", {}).get("model_version", "unknown")
    except Exception as e:
        logger.warning("could not read versions.yaml: %s", e)

    return ModelBundle(
        classifier=classifier,
        tokenizer=tokenizer,
        device=device,
        calibration_T=T,
        calibration_loaded=cal_loaded,
        model_version=model_version,
        max_length=config["training"]["max_length"],
        mini_batch_size=config["api"]["mini_batch_size"],
    )


def get_model(request: Request) -> ModelBundle:
    bundle = getattr(request.app.state, "model_bundle", None)
    if bundle is None:
        raise HTTPException(status_code=503, detail="model not loaded")
    return bundle


def get_monitor(request: Request) -> PredictionMonitor:
    monitor = getattr(request.app.state, "monitor", None)
    if monitor is None:
        raise HTTPException(status_code=503, detail="monitor not initialized")
    return monitor
